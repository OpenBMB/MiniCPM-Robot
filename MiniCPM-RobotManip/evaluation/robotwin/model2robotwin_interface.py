# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""RoboTwin policy interface backed by the MiniCPM WebSocket client."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)
from evaluation.robotwin.unified_ee6d import (
    GRIPPER_THRESHOLD,
    MODEL_DIM,
    binarize_ee6d_grippers,
    pack_robotwin_state,
    robotwin_ee6d_to_env_action,
    unpack_robotwin_eef,
)


ROBOTWIN_EMBODIMENT_ID = 4
# Added to every per-task evaluation step limit from the RoboTwin checkout.
DEFAULT_STEP_LIMIT_BONUS = 1000
STEP_LIMIT_FILENAME = "_eval_step_limit.yml"
PROMPT_TEMPLATE = (
    "The robot is RoboTwin2 ALOHA-AgileX, a simulated dual-arm ALOHA-style "
    "manipulator. Its action control method is absolute dual-arm end-effector "
    "pose in the unified 80D layout with gripper closed commands, and its "
    "action FPS is 15 Hz. Task: {instruction}"
)
# Pre-alignment wording, kept for the single-factor ablation only.
LEGACY_PROMPT_TEMPLATE = (
    "The robot is RoboTwin ALOHA-AgileX, a dual-arm ALOHA-style "
    "manipulator. Its action control method is absolute dual-arm end-effector pose "
    "in the unified 80D layout with gripper closed command, and its action FPS "
    "is 15 Hz. Task: {instruction}"
)
PROMPT_TEMPLATES = {
    "robotwin2": PROMPT_TEMPLATE,
    "legacy": LEGACY_PROMPT_TEMPLATE,
}
DEFAULT_CLIENT_IMAGE_SIZE = (448, 448)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _resolve_prompt_template() -> str:
    style = os.environ.get("ROBOTWIN_PROMPT_STYLE", "robotwin2").strip().lower()
    if style not in PROMPT_TEMPLATES:
        raise ValueError(
            "ROBOTWIN_PROMPT_STYLE must be one of "
            f"{sorted(PROMPT_TEMPLATES)}; got {style!r}"
        )
    return PROMPT_TEMPLATES[style]


def _apply_step_limit_bonus() -> None:
    """Add ``ROBOTWIN_EVAL_STEP_LIMIT_BONUS`` to every per-task step limit.

    RoboTwin reads ``_eval_step_limit.yml`` from ``envs._base_task.CONFIGS_PATH``
    lazily, per episode, so redirecting that attribute to a generated copy is
    enough; no RoboTwin source patch and no launcher change are needed. This
    runs inside the RoboTwin process because RoboTwin imports this policy module
    there. Set the bonus to ``0`` to use the checkout's limits unchanged.
    """

    bonus = int(
        os.environ.get("ROBOTWIN_EVAL_STEP_LIMIT_BONUS", str(DEFAULT_STEP_LIMIT_BONUS))
    )
    if bonus == 0:
        return

    try:
        import envs._base_task as base_task
    except ImportError:
        # Not running inside RoboTwin (unit tests, offline tooling).
        print("*** RoboTwin envs not importable; step limits left unchanged ***")
        return

    import yaml

    source = Path(base_task.CONFIGS_PATH) / STEP_LIMIT_FILENAME
    limits = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(limits, dict) or not limits:
        raise ValueError(f"{source} did not contain a task -> step limit mapping")

    target = Path(tempfile.mkdtemp(prefix="minicpm_robotwin_step_limit_"))
    atexit.register(shutil.rmtree, target, True)
    (target / STEP_LIMIT_FILENAME).write_text(
        yaml.safe_dump(
            {task: int(limit) + bonus for task, limit in limits.items()},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    # Only the step-limit lookup uses this attribute; the embodiment and camera
    # configs are read through the separate `envs.CONFIGS_PATH` import.
    base_task.CONFIGS_PATH = f"{target}{os.sep}"
    print(
        f"*** RoboTwin eval step limits +{bonus} "
        f"({len(limits)} tasks, from {source}) ***"
    )


class ModelClient:
    """Adapt RoboTwin observations to MiniCPM's stateless policy protocol."""

    def __init__(
        self,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        image_size: Sequence[int] | None = DEFAULT_CLIENT_IMAGE_SIZE,
        host: str = "127.0.0.1",
        port: int = 10093,
        action_mode: str = "abs",
        gripper_threshold: float = GRIPPER_THRESHOLD,
        gripper_close_position: float = 0.0,
        use_joint_state: bool = False,
        chain_eef: bool = False,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        if action_mode != "abs":
            raise ValueError(
                "MiniCPM RoboTwin evaluation only supports action_mode='abs'; "
                f"got {action_mode!r}"
            )
        # The reference evaluation client resizes on the client with cv2 INTER_AREA before the wire,
        # so this client does the same by default. The MiniCPM server still applies
        # the checkpoint's PIL 448x448 resize, which is an identity resample at
        # this size. Pass image_size=None to send simulator-resolution frames and
        # let the server own the only resize.
        if image_size is not None:
            if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
                raise ValueError(
                    f"image_size must contain two positive values, got {image_size!r}"
                )
            image_size = (int(image_size[0]), int(image_size[1]))
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError(f"horizon must be a non-negative integer, got {horizon!r}")

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.action_mode = action_mode
        self.image_size = image_size
        self.horizon = horizon
        self.gripper_threshold = float(gripper_threshold)
        self.gripper_close_position = float(gripper_close_position)
        self.use_joint_state = bool(use_joint_state)
        self.chain_eef = bool(chain_eef)
        self.prompt_template = prompt_template
        self.task_description: str | None = None
        self.image_history: deque[Any] = deque(maxlen=horizon)
        self.num_image_history = 0
        self.raw_actions: np.ndarray | None = None
        self._chained_ee6d: np.ndarray | None = None
        # Policy-step counter owned by this client. It is deliberately not
        # RoboTwin's take_action_cnt: the two diverge as soon as a model target
        # is submitted to the simulator more than once.
        self.control_step = 0

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
            f"image_size: {self.image_size}, use_joint_state: {self.use_joint_state}, "
            f"chain_eef: {self.chain_eef}, server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.num_image_history = 0
        self.raw_actions = None
        self._chained_ee6d = None
        self.control_step = 0

    def step(self, example: dict[str, Any], step: int = 0) -> np.ndarray:
        """Return the 16D absolute EE command for policy step ``step``.

        ``step`` counts policy steps, not simulator steps.
        """

        task_description = example.get("lang")
        if not isinstance(task_description, str):
            raise TypeError("example['lang'] must be a string")

        images = example.get("image")
        if not isinstance(images, (list, tuple)) or len(images) != 3:
            raise ValueError("example['image'] must contain exactly three images")

        if task_description != self.task_description:
            self.reset(task_description)

        if self.raw_actions is None or step % self.action_chunk_size == 0:
            prepared_images = [self._prepare_image(image) for image in images]
            joint_state = np.asarray(example["joint_state"], dtype=np.float32).reshape(-1)
            if not self.use_joint_state:
                # Joint channels stay zero; only the EEF channels are populated.
                joint_state = np.zeros_like(joint_state)
            state = pack_robotwin_state(
                joint_state,
                example["endpose"],
                ee6d_override=self._chained_ee6d if self.chain_eef else None,
            )
            model_example = {
                "lang": self.prompt_template.format(instruction=task_description),
                "image": prepared_images,
                "state": state[None],
            }
            response = self.client.predict_action(
                {
                    "examples": [model_example],
                    "embodiment_id": ROBOTWIN_EMBODIMENT_ID,
                }
            )
            self.raw_actions = self._validate_response(response)

        action_idx = step % self.action_chunk_size
        active_action = unpack_robotwin_eef(self.raw_actions[action_idx])

        # Ablation path only. When chaining is enabled the next chunk boundary
        # reports the last absolute command instead of the measured endpose,
        # with both gripper channels hard 0/1 rather than regressed.
        if self.chain_eef:
            self._chained_ee6d = binarize_ee6d_grippers(
                active_action, threshold=self.gripper_threshold
            )
        return robotwin_ee6d_to_env_action(
            active_action,
            threshold=self.gripper_threshold,
            gripper_close_position=self.gripper_close_position,
        )

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
        if raw_actions.shape[1] != MODEL_DIM:
            raise ValueError(
                f"Policy action dimension must be {MODEL_DIM}; "
                f"got {raw_actions.shape[1]}"
            )
        if not np.issubdtype(raw_actions.dtype, np.floating):
            raise TypeError(f"Policy actions must be floating, got {raw_actions.dtype}")
        if not np.isfinite(raw_actions).all():
            raise ValueError("Policy actions contain non-finite values")
        return raw_actions

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
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
        if self.image_size is None:
            return np.ascontiguousarray(image)

        import cv2  # Default path; matches the reference client-side resize.

        return cv2.resize(
            image,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_AREA,
        )


def get_model(usr_args: dict[str, Any]) -> ModelClient:
    """RoboTwin policy factory."""

    action_mode = usr_args.get("action_mode", "abs")
    if action_mode != "abs":
        raise ValueError(
            "MiniCPM RoboTwin evaluation only supports action_mode='abs'; "
            f"got {action_mode!r}"
        )
    client_resize = _env_bool("ROBOTWIN_CLIENT_RESIZE", True)
    _apply_step_limit_bonus()
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 10093)),
        action_mode=action_mode,
        image_size=DEFAULT_CLIENT_IMAGE_SIZE if client_resize else None,
        gripper_threshold=_env_float(
            "ROBOTWIN_GRIPPER_CLOSED_THRESHOLD", GRIPPER_THRESHOLD
        ),
        gripper_close_position=_env_float("ROBOTWIN_GRIPPER_CLOSE_POSITION", 0.0),
        use_joint_state=_env_bool("ROBOTWIN_STATE_JOINTS", False),
        chain_eef=_env_bool("ROBOTWIN_CHAIN_EEF", False),
        prompt_template=_resolve_prompt_template(),
    )


def reset_model(model: ModelClient) -> None:
    """Reset action-chunk and chained-EEF state between RoboTwin episodes."""

    model.reset(task_description="")


def eval(TASK_ENV: Any, model: ModelClient, observation: dict[str, Any]) -> None:
    """Run one RoboTwin policy step."""

    instruction = TASK_ENV.get_instruction()
    camera_observations = observation["observation"]
    endpose = observation["endpose"]

    # Training camera order: [head, left_wrist, right_wrist].
    images = [
        camera_observations["head_camera"]["rgb"],
        camera_observations["left_camera"]["rgb"],
        camera_observations["right_camera"]["rgb"],
    ]
    example = {
        "lang": str(instruction),
        "image": images,
        "joint_state": observation["joint_action"]["vector"],
        "endpose": np.concatenate(
            (
                np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(7),
                np.asarray([endpose["left_gripper"]], dtype=np.float32),
                np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(7),
                np.asarray([endpose["right_gripper"]], dtype=np.float32),
            )
        ),
    }
    action = model.step(example, step=model.control_step)
    model.control_step += 1
    TASK_ENV.take_action(action, action_type="ee")
