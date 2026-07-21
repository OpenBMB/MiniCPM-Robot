# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""LIBERO env-side adapter for the MiniCPM policy server.

The server returns execution-ready model actions. This client only keeps the
source evaluator's image resizing, action-chunk scheduling, optional ensemble
state, visualization, and LIBERO-specific seven-dimensional action slicing.
It performs no normalization or un-normalization.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)
from evaluation.common.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 10093,
        image_size: Sequence[int] = (448, 448),
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
        if actions.shape[1] <= 0 or actions.shape[2] < 7:
            raise ValueError(
                "Policy actions must have shape (1, T, D) with T > 0 and D >= 7; "
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
        """Run one environment step and return the first seven action dimensions."""
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
            # State is intentionally absent. The MiniCPM server uses its existing
            # 80-dimensional zero-state behavior for standard LIBERO evaluation.
            vla_input = {"examples": [example]}
            response = self.client.predict_action(vla_input)
            actions_batch = self._validate_actions_response(response)
            self.raw_actions = actions_batch[0]

        # Preserve model values here; LIBERO-only gripper conversion happens in
        # eval_libero.py immediately before env.step().
        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
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
