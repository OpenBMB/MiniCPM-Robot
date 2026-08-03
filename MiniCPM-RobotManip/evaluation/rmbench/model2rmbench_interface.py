# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMBench policy client backed by the memVLA WebSocket server.

The simulator speaks RMBench's native 14D joint layout; the checkpoint speaks the
unified 80D layout. This client owns that mapping in both directions -- the server
stays benchmark-agnostic and transports 80D only:

* ``pack_state``: 14D measured joints -> 80D. Left joints fill 0:6, right joints
  17:23; grippers (raw opening in [0, 1], larger is more open) binarise to
  1 = closed at 0.4; every other channel is zero.
* ``decode_action``: 80D prediction -> 14D command. Extract the same joint slots;
  the gripper-closed prediction binarises at 0.5 and converts back to an opening.

It also owns the per-camera history buffer, so the caller only observes frames and
asks for one action chunk per decision.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from deployment.model_server import memvla_recipe as recipe
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from evaluation.rmbench.history_buffer import HistoryBuffer


def pack_state(state14: Sequence[float]) -> np.ndarray:
    """Pack a measured 14D dual-arm joint vector into the unified 80D layout."""
    state14 = np.asarray(state14, dtype=np.float32).reshape(-1)
    if state14.size != recipe.SIM_ACTION_DIM:
        raise ValueError(f"state must contain {recipe.SIM_ACTION_DIM} values, got {state14.size}")

    packed = np.zeros(recipe.UNIFIED_DIM, dtype=np.float32)
    ls, le = recipe.LEFT_JOINT_SLICE
    rs, re = recipe.RIGHT_JOINT_SLICE
    sls, sle = recipe.SIM_LEFT_JOINT_SLICE
    srs, sre = recipe.SIM_RIGHT_JOINT_SLICE
    packed[ls:le] = state14[sls:sle]
    packed[rs:re] = state14[srs:sre]
    # Gripper closed mode: opening > 0.4 counts as open, so closed = 0.
    packed[recipe.LEFT_GRIPPER_INDEX] = float(
        state14[recipe.SIM_LEFT_GRIPPER_INDEX] <= recipe.GRIPPER_OPEN_THRESHOLD
    )
    packed[recipe.RIGHT_GRIPPER_INDEX] = float(
        state14[recipe.SIM_RIGHT_GRIPPER_INDEX] <= recipe.GRIPPER_OPEN_THRESHOLD
    )
    return packed


def decode_action(action80: Sequence[float]) -> np.ndarray:
    """Convert one unified 80D action into RMBench's 14D absolute joint command."""
    action80 = np.asarray(action80, dtype=np.float32).reshape(-1)
    if action80.size != recipe.UNIFIED_DIM:
        raise ValueError(f"action must contain {recipe.UNIFIED_DIM} values, got {action80.size}")

    command = np.zeros(recipe.SIM_ACTION_DIM, dtype=np.float32)
    ls, le = recipe.LEFT_JOINT_SLICE
    rs, re = recipe.RIGHT_JOINT_SLICE
    sls, sle = recipe.SIM_LEFT_JOINT_SLICE
    srs, sre = recipe.SIM_RIGHT_JOINT_SLICE
    command[sls:sle] = action80[ls:le]
    command[srs:sre] = action80[rs:re]
    # 1 = closed prediction -> opening 0; otherwise opening 1.
    command[recipe.SIM_LEFT_GRIPPER_INDEX] = float(
        action80[recipe.LEFT_GRIPPER_INDEX] < recipe.GRIPPER_CLOSED_THRESHOLD
    )
    command[recipe.SIM_RIGHT_GRIPPER_INDEX] = float(
        action80[recipe.RIGHT_GRIPPER_INDEX] < recipe.GRIPPER_CLOSED_THRESHOLD
    )
    return command


class RMBenchPolicyClient:
    """Drive one RMBench rollout through the memVLA WebSocket server."""

    def __init__(
        self,
        camera_keys: Sequence[str],
        host: str = "127.0.0.1",
        port: int = 10094,
        response_timeout: float = 300.0,
        max_message_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.client = WebsocketClientPolicy(
            host,
            port,
            response_timeout=response_timeout,
            max_message_bytes=max_message_bytes,
        )
        metadata = self.client.get_server_metadata()
        capabilities = metadata.get("capabilities", {})
        if not capabilities.get("history") or not capabilities.get("subtask"):
            self.client.close()
            raise RuntimeError(
                "RMBench evaluation requires a memVLA history/subtask server; the connected "
                f"server does not advertise it (capabilities={capabilities})"
            )
        action_chunk_size = metadata.get("action_chunk_size")
        if not isinstance(action_chunk_size, (int, np.integer)) or int(action_chunk_size) <= 0:
            self.client.close()
            raise RuntimeError("Server metadata must contain a positive integer action_chunk_size")
        self.action_chunk_size = int(action_chunk_size)
        self.embodiment_id = int(metadata.get("default_embodiment_id", recipe.EMBODIMENT_ID))
        self.buffer = HistoryBuffer(camera_keys)
        self.last_subtask = ""

    def reset(self) -> None:
        self.buffer.reset()
        self.last_subtask = ""

    def observe(self, frames: dict[str, np.ndarray]) -> None:
        self.buffer.observe(frames)

    def predict_chunk(self, instruction: str, state14: Sequence[float]) -> np.ndarray:
        """Return one ``(action_chunk_size, 14)`` chunk of absolute joint targets."""
        views = self.buffer.build_views()
        example = {
            "lang": str(instruction),
            "views": views,
            "state": pack_state(state14)[None],
        }
        response = self.client.predict_action(
            {"examples": [example], "embodiment_id": self.embodiment_id}
        )
        actions80, subtask = self._validate_response(response)
        self.last_subtask = subtask
        return np.stack([decode_action(actions80[t]) for t in range(actions80.shape[0])], axis=0)

    def _validate_response(self, response: Any) -> tuple[np.ndarray, str]:
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
                f"Policy actions must have shape (1, action_chunk_size, D); got {actions.shape}"
            )
        chunk = actions[0]
        if chunk.shape[0] != self.action_chunk_size:
            raise ValueError(
                f"Policy action chunk does not match metadata: expected {self.action_chunk_size}, "
                f"got {chunk.shape[0]}"
            )
        if chunk.shape[1] != recipe.UNIFIED_DIM:
            raise ValueError(f"Policy action dimension must be {recipe.UNIFIED_DIM}; got {chunk.shape[1]}")
        if not np.issubdtype(chunk.dtype, np.floating):
            raise TypeError(f"Policy actions must be floating, got {chunk.dtype}")
        if not np.isfinite(chunk).all():
            raise ValueError("Policy actions contain non-finite values")
        subtask = data.get("subtask", "")
        return chunk, str(subtask)

    def close(self) -> None:
        self.client.close()
