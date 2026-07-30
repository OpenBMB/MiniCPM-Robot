# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CALVIN unified-80D EE6D adapter for the MiniCPM policy server."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)


MODEL_DIM = 80
CALVIN_EEF_SLICE = slice(7, 17)
CALVIN_EEF_DIM = 10
ACTION_CHUNK_SIZE = 30
GRIPPER_THRESHOLD = 0.5
_EPS = 1e-8


def pack_calvin_state(state: object) -> np.ndarray:
    """Pack xyz + interleaved rotation6D + gripper_closed into ``(1, 80)``."""

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != CALVIN_EEF_DIM:
        raise ValueError(
            f"CALVIN EE6D state must contain {CALVIN_EEF_DIM} values; "
            f"got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("CALVIN EE6D state must contain only finite values")
    packed = np.zeros((1, MODEL_DIM), dtype=np.float32)
    packed[0, CALVIN_EEF_SLICE] = values
    return packed


def interleaved_rotation6d_to_matrix(rotation6d: object) -> np.ndarray:
    """Convert ``matrix[:, :2].reshape(6)`` values to an orthonormal matrix."""

    values = np.asarray(rotation6d, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError(f"rotation6D must contain 6 values; got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("rotation6D must contain only finite values")

    first = values[0::2]
    second = values[1::2]
    first_norm = np.linalg.norm(first)
    if first_norm < _EPS:
        raise ValueError("rotation6D first column must be nonzero")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < _EPS:
        raise ValueError("rotation6D columns must not be collinear")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def calvin_ee6d_to_action(values: object) -> dict[str, object]:
    """Decode one physical 10D EE6D target as a CALVIN absolute action."""

    action = np.asarray(values, dtype=np.float64).reshape(-1)
    if action.size != CALVIN_EEF_DIM:
        raise ValueError(
            f"CALVIN EE6D action must contain {CALVIN_EEF_DIM} values; "
            f"got {action.size}"
        )
    if not np.isfinite(action).all():
        raise ValueError("CALVIN EE6D action must contain only finite values")

    euler = Rotation.from_matrix(
        interleaved_rotation6d_to_matrix(action[3:9])
    ).as_euler("xyz")
    gripper = -1.0 if float(action[9]) > GRIPPER_THRESHOLD else 1.0
    env_action = np.concatenate((action[:3], euler, [gripper])).astype(np.float32)
    return {"action": env_action, "type": "cartesian_abs"}


class ModelClient:
    """Transport, chunk cache, and action decoder for CALVIN unified-80D."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        image_size: Sequence[int] = (448, 448),
        embodiment_id: int = 1,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata()
        expected = {
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "action_dim": MODEL_DIM,
            "state_dim": MODEL_DIM,
            "action_normalization": "none",
            "actions_ready_for_execution": True,
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatched:
            self.client.close()
            details = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in mismatched.items()
            )
            raise ValueError(f"CALVIN policy metadata mismatch: {details}")

        if isinstance(embodiment_id, bool) or not isinstance(
            embodiment_id, (int, np.integer)
        ):
            self.client.close()
            raise TypeError("embodiment_id must be an integer")
        self.embodiment_id = int(embodiment_id)
        max_num_embodiments = metadata.get("max_num_embodiments")
        valid_range = self.embodiment_id >= 0
        if max_num_embodiments is not None:
            valid_range = valid_range and (
                isinstance(max_num_embodiments, int)
                and not isinstance(max_num_embodiments, bool)
                and self.embodiment_id < max_num_embodiments
            )
        if not valid_range:
            self.client.close()
            raise ValueError(
                "embodiment_id must be within the server's embodiment range; "
                f"got {self.embodiment_id}, max_num_embodiments="
                f"{max_num_embodiments!r}"
            )

        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.image_size
        ):
            self.client.close()
            raise ValueError(
                f"image_size must contain two positive integers, got {image_size!r}"
            )

        self.action_chunk_size = ACTION_CHUNK_SIZE
        self.task_description: Optional[str] = None
        self.raw_actions: Optional[np.ndarray] = None

    def reset(self, task_description: Optional[str] = None) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def _resize_images(self, images: object) -> list[np.ndarray]:
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            count = len(images) if isinstance(images, (list, tuple)) else 0
            raise ValueError(
                f"CALVIN unified EE6D requires exactly 2 images; got {count}"
            )

        resized = []
        for index, image in enumerate(images):
            array = np.asarray(image)
            if array.dtype != np.uint8:
                raise ValueError(
                    f"CALVIN image {index} must have dtype uint8; got {array.dtype}"
                )
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(
                    f"CALVIN image {index} must have shape HxWx3; got {array.shape}"
                )
            if array.shape[:2] != self.image_size:
                array = np.asarray(
                    Image.fromarray(array).resize(
                        (self.image_size[1], self.image_size[0])
                    )
                )
            resized.append(np.ascontiguousarray(array))
        return resized

    @staticmethod
    def _validate_response(response: object) -> np.ndarray:
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
        if not isinstance(data, dict) or "actions" not in data:
            raise KeyError("Policy response data must contain 'actions'")

        actions = np.asarray(data["actions"])
        if actions.shape != (1, ACTION_CHUNK_SIZE, MODEL_DIM):
            raise ValueError(
                "CALVIN policy actions must have shape "
                f"(1, {ACTION_CHUNK_SIZE}, {MODEL_DIM}); got {actions.shape}"
            )
        if not np.issubdtype(actions.dtype, np.floating):
            raise TypeError(
                f"CALVIN policy actions must have a floating dtype; got {actions.dtype}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("CALVIN policy actions must contain only finite values")
        return actions.astype(np.float32, copy=False)

    def step(self, example: dict, step: int = 0) -> dict[str, object]:
        """Return one CALVIN absolute action, refreshing every 30 env steps."""

        task_description = example.get("lang")
        if task_description != self.task_description:
            self.reset(task_description=task_description)

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            if example.get("state") is None:
                raise KeyError("CALVIN unified EE6D requires example['state']")
            wire_example = {
                "image": self._resize_images(example.get("image")),
                "lang": task_description,
                "state": pack_calvin_state(example["state"]),
            }
            response = self.client.predict_action(
                {
                    "examples": [wire_example],
                    "embodiment_id": self.embodiment_id,
                }
            )
            self.raw_actions = self._validate_response(response)[0]

        active = self.raw_actions[step % self.action_chunk_size, CALVIN_EEF_SLICE]
        return calvin_ee6d_to_action(active)
