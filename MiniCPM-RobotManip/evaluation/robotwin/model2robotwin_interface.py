# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""RoboTwin policy interface backed by the MiniCPM WebSocket client."""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)


ROBOTWIN_ACTION_DIM = 14
ACTION_REORDER = (0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13)


class ModelClient:
    """Adapt RoboTwin observations to MiniCPM's stateless policy protocol."""

    def __init__(
        self,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        image_size: Sequence[int] = (448, 448),
        host: str = "127.0.0.1",
        port: int = 10093,
        action_mode: str = "abs",
    ) -> None:
        if action_mode != "abs":
            raise ValueError(
                "MiniCPM RoboTwin evaluation only supports action_mode='abs'; "
                f"got {action_mode!r}"
            )
        if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
            raise ValueError(
                f"image_size must contain two positive values, got {image_size!r}"
            )
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError(f"horizon must be a non-negative integer, got {horizon!r}")

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.action_mode = action_mode
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.horizon = horizon
        self.task_description: str | None = None
        self.image_history: deque[Any] = deque(maxlen=horizon)
        self.num_image_history = 0
        self.raw_actions: np.ndarray | None = None

        server_meta = self.client.get_server_metadata()
        action_chunk_size = server_meta.get("action_chunk_size")
        if (
            isinstance(action_chunk_size, bool)
            or not isinstance(action_chunk_size, (int, np.integer))
            or int(action_chunk_size) <= 0
        ):
            self.client.close()
            raise RuntimeError(
                "Server metadata must contain a positive integer action_chunk_size"
            )
        self.action_chunk_size = int(action_chunk_size)
        print(
            f"*** policy_setup: {policy_setup}, action_mode: {action_mode}, "
            f"image_size: {self.image_size}, server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.num_image_history = 0
        self.raw_actions = None

    def step(self, example: dict[str, Any], step: int = 0) -> np.ndarray:
        task_description = example.get("lang")
        if not isinstance(task_description, str):
            raise TypeError("example['lang'] must be a string")

        images = example.get("image")
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("example['image'] must be a non-empty image sequence")

        if task_description != self.task_description:
            self.reset(task_description)

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            resized_images = [self._resize_image(image) for image in images]
            # RoboTwin exposes a 14-D joint vector, while MiniCPM was trained with
            # an 80-D state. Keep state absent so the server supplies zero80.
            model_example = {
                "lang": task_description,
                "image": resized_images,
            }
            response = self.client.predict_action({"examples": [model_example]})
            self.raw_actions = self._validate_response(response)

        action_idx = step % self.action_chunk_size
        current_action = self.raw_actions[action_idx]
        return current_action[list(ACTION_REORDER)]

    def _validate_response(self, response: Any) -> np.ndarray:
        if not isinstance(response, dict):
            raise RuntimeError("Policy response must be a dict")
        if response.get("ok") is not True:
            raise RuntimeError(f"Policy request failed: {response.get('error', response)!r}")

        data = response.get("data")
        if not isinstance(data, dict) or "actions" not in data:
            raise RuntimeError("Policy response is missing data.actions")

        actions = np.asarray(data["actions"])
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(
                "Policy actions must have shape (1, action_chunk_size, D); "
                f"got {actions.shape}"
            )

        raw_actions = actions[0]
        if raw_actions.shape[0] != self.action_chunk_size:
            raise ValueError(
                "Policy action chunk does not match server metadata: "
                f"expected {self.action_chunk_size}, got {raw_actions.shape[0]}"
            )
        if raw_actions.shape[1] < ROBOTWIN_ACTION_DIM:
            raise ValueError(
                f"Policy action dimension must be at least {ROBOTWIN_ACTION_DIM}; "
                f"got {raw_actions.shape[1]}"
            )
        if not np.issubdtype(raw_actions.dtype, np.floating):
            raise TypeError(f"Policy actions must be floating, got {raw_actions.dtype}")
        if not np.isfinite(raw_actions).all():
            raise ValueError("Policy actions contain non-finite values")
        return raw_actions

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"RoboTwin camera image must be a NumPy array, got {type(image)!r}"
            )
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"RoboTwin camera image must have shape HxWx3, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"RoboTwin camera image must have dtype uint8, got {image.dtype}"
            )
        return np.asarray(
            Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0])
            )
        )


def get_model(usr_args: dict[str, Any]) -> ModelClient:
    """RoboTwin policy factory."""

    action_mode = usr_args.get("action_mode", "abs")
    if action_mode != "abs":
        raise ValueError(
            "MiniCPM RoboTwin evaluation only supports action_mode='abs'; "
            f"got {action_mode!r}"
        )
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 10093)),
        action_mode=action_mode,
    )


def reset_model(model: ModelClient) -> None:
    """Reset action-chunk state between RoboTwin episodes."""

    model.reset(task_description="")


def eval(TASK_ENV: Any, model: ModelClient, observation: dict[str, Any]) -> None:
    """Run one RoboTwin policy step."""

    instruction = TASK_ENV.get_instruction()
    camera_observations = observation["observation"]

    # Training camera order: [head, left_wrist, right_wrist].
    images = [
        camera_observations["head_camera"]["rgb"],
        camera_observations["left_camera"]["rgb"],
        camera_observations["right_camera"]["rgb"],
    ]
    example = {
        "lang": str(instruction),
        "image": images,
    }
    action = model.step(example, step=TASK_ENV.take_action_cnt)
    TASK_ENV.take_action(action)
