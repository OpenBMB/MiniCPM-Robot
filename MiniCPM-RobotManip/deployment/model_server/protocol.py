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

"""Wire-level contracts for stateless and future streaming policy calls."""

from __future__ import annotations

from typing import Any, Protocol


PROTOCOL_VERSION = 1

PING = "ping"
INFER = "infer"
PREDICT_ACTION = "predict_action"

SESSION_OPEN = "session.open"
STREAM_INFER = "stream.infer"
SESSION_RESET = "session.reset"
SESSION_CLOSE = "session.close"
RESERVED_STREAM_MESSAGE_TYPES = (
    SESSION_OPEN,
    STREAM_INFER,
    SESSION_RESET,
    SESSION_CLOSE,
)


class ProtocolError(ValueError):
    """Raised when a decoded MessagePack frame violates the wire contract."""


class FramePolicy(Protocol):
    """Current stateless model boundary."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def predict_action(self, **payload: Any) -> dict[str, Any]: ...


class StreamingPolicy(Protocol):
    """Optional future model boundary; MiniCPM's current adapter does not implement it."""

    def open_session(self, **payload: Any) -> dict[str, Any]: ...

    def predict_stream_frame(self, **payload: Any) -> dict[str, Any]: ...

    def reset_session(self, **payload: Any) -> dict[str, Any]: ...

    def close_session(self, **payload: Any) -> dict[str, Any]: ...


def build_server_metadata(
    policy_metadata: dict[str, Any],
    *,
    checkpoint: str,
) -> dict[str, Any]:
    """Build the first frame sent to every connected starVLA client."""
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "server": "minicpm_robot_manip",
        "env": "minicpm_robot_manip",
        "ckpt_path": checkpoint,
        "capabilities": {
            "infer": True,
            "single_frame": True,
            "multi_view": True,
            "streaming": False,
            "sessions": False,
            "reserved_message_types": list(RESERVED_STREAM_MESSAGE_TYPES),
        },
    }
    metadata.update(policy_metadata)
    return metadata


def decode_request(message: Any) -> tuple[str, Any, dict[str, Any]]:
    """Normalize starVLA flat payloads and versioned request envelopes."""
    if not isinstance(message, dict):
        raise ProtocolError("Request must decode to a dict")

    request_id = message.get("request_id", "default")
    message_type = message.get("type", INFER)
    if not isinstance(message_type, str) or not message_type:
        raise ProtocolError("Request type must be a non-empty string")

    if "payload" in message:
        payload = message["payload"]
    else:
        payload = {
            key: value
            for key, value in message.items()
            if key not in {"type", "request_id"}
        }
    if not isinstance(payload, dict):
        raise ProtocolError("Request payload must be a dict")
    return message_type, request_id, payload


def success_response(
    *,
    request_id: Any,
    message_type: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": "ok",
        "ok": True,
        "type": message_type,
        "request_id": request_id,
    }
    if data is not None:
        response["data"] = data
    return response


def error_response(
    *,
    request_id: Any = "default",
    message: str,
    code: str,
    message_type: str = "error",
) -> dict[str, Any]:
    return {
        "status": "error",
        "ok": False,
        "type": message_type,
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
