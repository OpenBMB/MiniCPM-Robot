# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Small synchronous client for the MiniCPM/starVLA wire protocol."""

from __future__ import annotations

import inspect
from typing import Any

import websockets.sync.client

from deployment.model_server.tools import msgpack_numpy


class WebsocketClientPolicy:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 10093,
        *,
        open_timeout: float = 30,
        response_timeout: float | None = 30,
        max_message_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        uri = f"ws://{host}"
        if port is not None:
            uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        connect_kwargs = {
            "compression": None,
            "max_size": max_message_bytes,
            "open_timeout": open_timeout,
            "ping_interval": None,
        }
        # websockets 15 added the proxy argument. Evaluator environments may use
        # websockets 13 on Python 3.8, so disable proxies only when supported.
        if "proxy" in inspect.signature(
            websockets.sync.client.connect
        ).parameters:
            connect_kwargs["proxy"] = None
        self._ws = websockets.sync.client.connect(uri, **connect_kwargs)
        self._response_timeout = response_timeout
        metadata_frame = self._ws.recv(timeout=open_timeout)
        if isinstance(metadata_frame, str):
            self._ws.close()
            raise RuntimeError("Server metadata must be a binary MessagePack frame")
        metadata = msgpack_numpy.unpackb(metadata_frame)
        if not isinstance(metadata, dict):
            self._ws.close()
            raise RuntimeError("Server metadata must decode to a dict")
        self._server_metadata = metadata

    def __enter__(self) -> "WebsocketClientPolicy":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def _send_receive(self, message: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(message))
        frame = self._ws.recv(timeout=self._response_timeout)
        if isinstance(frame, str):
            raise RuntimeError(f"Server returned an unexpected text frame: {frame}")
        response = msgpack_numpy.unpackb(frame)
        if not isinstance(response, dict):
            raise RuntimeError("Server response must decode to a dict")
        return response

    def predict_action(self, query_info: dict[str, Any]) -> dict[str, Any]:
        """Send the flat payload used by existing starVLA evaluators."""
        return self._send_receive(query_info)

    def request(
        self,
        message_type: str,
        payload: dict[str, Any] | None = None,
        *,
        request_id: Any = "default",
    ) -> dict[str, Any]:
        return self._send_receive(
            {
                "type": message_type,
                "request_id": request_id,
                "payload": payload or {},
            }
        )

    def ping(self, *, request_id: Any = "default") -> dict[str, Any]:
        return self.request("ping", request_id=request_id)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
