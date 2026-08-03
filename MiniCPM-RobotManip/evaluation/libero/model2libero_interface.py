# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""LIBERO unified-80D EE6D adapter for the MiniCPM policy server."""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)
from evaluation.common.adaptive_ensemble import AdaptiveEnsembler


MODEL_DIM = 80
LIBERO_EEF_SLICE = slice(7, 17)
LIBERO_EEF_DIM = 10
GRIPPER_THRESHOLD = 0.5


def _pack_libero_state(state: object) -> np.ndarray:
    """Insert xyz + contiguous-column rotation6D + gripper_closed into 80D."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != LIBERO_EEF_DIM:
        raise ValueError(
            f"LIBERO EE6D state must contain {LIBERO_EEF_DIM} values; "
            f"got {values.size}"
        )
    packed = np.zeros(MODEL_DIM, dtype=np.float32)
    packed[LIBERO_EEF_SLICE] = values
    return packed


def _rotation6d_to_matrix(rotation6d: object) -> np.ndarray:
    """Convert contiguous rotation-matrix columns to an orthonormal matrix."""
    values = np.asarray(rotation6d, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError(f"rotation6D must contain 6 values; got {values.size}")
    first = values[:3]
    second = values[3:6]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("rotation6D first column must be nonzero")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError("rotation6D columns must not be collinear")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


class ModelClient:
    def __init__(
        self,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 20000,
        image_size: Sequence[int] = (448, 448),
        unified_ee6d: bool = False,
    ) -> None:
        # Connect and receive model-invariant handshake metadata.
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        action_chunk_size = meta.get("action_chunk_size")
        if (
            isinstance(action_chunk_size, bool)
            or not isinstance(action_chunk_size, (int, np.integer))
            or int(action_chunk_size) <= 0
        ):
            self.client.close()
            raise ValueError(
                "Server metadata action_chunk_size must be a positive integer; "
                f"got {action_chunk_size!r}"
            )
        self.action_chunk_size = int(action_chunk_size)
        self._server_metadata = meta

        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.image_size
        ):
            self.client.close()
            raise ValueError(
                f"image_size must contain two positive integers, got {image_size!r}"
            )
        self.policy_setup = policy_setup
        self.unified_ee6d = unified_ee6d
        print(
            f"*** policy_setup: {policy_setup}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state is retained for parity with the source adapter.
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached execution-ready model chunk, refreshed every action_chunk_size steps.
        self.raw_actions: Optional[np.ndarray] = None
        self._chained_state: Optional[np.ndarray] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None
        self._chained_state = None

    def _validate_actions_response(self, response: object) -> np.ndarray:
        if not isinstance(response, dict):
            raise TypeError(
                f"Policy response must be a dict, got {type(response).__name__}"
            )
        if response.get("ok") is not True:
            raise RuntimeError(
                "Policy response did not report ok=true: "
                f"{response.get('error', response)!r}"
            )

        data = response.get("data")
        if not isinstance(data, dict):
            raise TypeError(
                f"Policy response data must be a dict, got {type(data).__name__}"
            )
        if "actions" not in data:
            raise KeyError(
                f"Key 'actions' not found in response data: keys={list(data)}"
            )

        actions = np.asarray(data["actions"])
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(
                "Policy actions must have shape (1, T, D); "
                f"got {actions.shape}"
            )
        minimum_action_dim = LIBERO_EEF_SLICE.stop if self.unified_ee6d else 7
        if actions.shape[1] <= 0 or actions.shape[2] < minimum_action_dim:
            raise ValueError(
                "Policy actions must have shape (1, T, D) with T > 0 and "
                f"D >= {minimum_action_dim}; "
                f"got {actions.shape}"
            )
        if actions.shape[1] != self.action_chunk_size:
            raise ValueError(
                "Policy action horizon disagrees with server metadata: "
                f"actions T={actions.shape[1]}, "
                f"action_chunk_size={self.action_chunk_size}"
            )
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError(
                f"Policy actions must have a floating dtype, got {actions.dtype}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Policy actions must contain only finite values")
        return actions

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """Return one absolute LIBERO OSC target decoded from unified channels 7:17."""
        del kwargs
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Resize both synchronized camera views to MiniCPM's training resolution.
        if self.image_size and example.get("image"):
            resized = []
            target_hw = self.image_size
            for img in example["image"]:
                arr = np.asarray(img)
                if arr.shape[:2] != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0])
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}

        # Refresh the chunk when its cached actions have been consumed.
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            wire_example = example
            if self.unified_ee6d:
                state = self._chained_state
                if state is None:
                    state = example.get("state")
                wire_example = {**example, "state": _pack_libero_state(state)}
            vla_input = {"examples": [wire_example]}
            response = self.client.predict_action(vla_input)
            actions_batch = self._validate_actions_response(response)
            self.raw_actions = actions_batch[0]

        if not self.unified_ee6d:
            raw_action = self.raw_actions[step % self.action_chunk_size]
            return {
                "raw_action": {
                    "world_vector": raw_action[:3].copy(),
                    "rotation_delta": raw_action[3:6].copy(),
                    "open_gripper": raw_action[6:7].copy(),
                }
            }

        active_action = np.asarray(
            self.raw_actions[step % self.action_chunk_size][LIBERO_EEF_SLICE],
            dtype=np.float32,
        )
        self._chained_state = active_action.copy()
        self._chained_state[9] = (
            1.0 if float(active_action[9]) > GRIPPER_THRESHOLD else 0.0
        )
        axis_angle = Rotation.from_matrix(
            _rotation6d_to_matrix(active_action[3:9])
        ).as_rotvec()
        gripper = np.asarray(
            [1.0 if float(active_action[9]) > GRIPPER_THRESHOLD else -1.0],
            dtype=np.float32,
        )
        raw_action = {
            "world_vector": active_action[:3].copy(),
            "rotation_delta": axis_angle.astype(np.float32),
            "open_gripper": gripper,
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        action_dim_labels = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(action_dim_labels), action_dim_labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate(
                    [
                        action["world_vector"],
                        action["rotation_delta"],
                        action["open_gripper"],
                    ],
                    axis=-1,
                )
                for action in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(action_dim_labels):
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
