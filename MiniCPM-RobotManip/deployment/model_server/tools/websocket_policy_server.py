# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Async MessagePack WebSocket transport for MiniCPM policies."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import websockets.asyncio.server
from websockets.exceptions import ConnectionClosed

from deployment.model_server import protocol
from deployment.model_server.tools import msgpack_numpy


RouteHandler = Callable[
    [str, Any, dict[str, Any]],
    Awaitable[dict[str, Any]],
]

_CORE_MESSAGE_TYPES = frozenset(
    {
        protocol.PING,
        protocol.INFER,
        protocol.PREDICT_ACTION,
    }
)


class WebsocketPolicyServer:
    """Serve a frame policy with starVLA-compatible transport semantics."""

    def __init__(
        self,
        policy: protocol.FramePolicy,
        host: str = "127.0.0.1",
        port: int = 10093,
        idle_timeout: int = -1,
        metadata: dict[str, Any] | None = None,
        max_message_bytes: int = 16 * 1024 * 1024,
        extra_handlers: Mapping[str, RouteHandler] | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError(f"port must be in [0, 65535], got {port}")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")

        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._idle_timeout = idle_timeout
        self._max_message_bytes = max_message_bytes
        self._last_active = time.monotonic()
        self._active_inferences = 0
        self._inference_lock = asyncio.Lock()
        self._server: Any = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None

        self._handlers: dict[str, RouteHandler] = {
            protocol.PING: self._handle_ping,
            protocol.INFER: self._handle_infer,
            protocol.PREDICT_ACTION: self._handle_infer,
        }
        for message_type in protocol.RESERVED_STREAM_MESSAGE_TYPES:
            self._handlers[message_type] = self._handle_unsupported_capability
        for message_type, handler in (extra_handlers or {}).items():
            self.register_handler(message_type, handler)

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @property
    def bound_port(self) -> int | None:
        """Actual listening port, including an OS-assigned port when configured as 0."""
        return self._bound_port

    def register_handler(self, message_type: str, handler: RouteHandler) -> None:
        """Register or replace a non-core route, including reserved stream routes."""
        if not isinstance(message_type, str) or not message_type:
            raise ValueError("message_type must be a non-empty string")
        if message_type in _CORE_MESSAGE_TYPES:
            raise ValueError(f"Cannot replace core message type {message_type!r}")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[message_type] = handler

    async def start(self) -> None:
        """Start listening without blocking; primarily useful for integration tests."""
        if self._server is not None:
            raise RuntimeError("WebSocket server is already running")
        self._last_active = time.monotonic()
        self._server = await websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_bytes,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self._bound_port = int(sockets[0].getsockname()[1])
        else:
            self._bound_port = self._port
        if self._idle_timeout > 0:
            self._watchdog_task = asyncio.create_task(self._idle_watchdog())
        logging.info(
            "WebSocket policy server listening on %s:%s",
            self._host,
            self._bound_port,
        )

    async def run(self) -> None:
        await self.start()
        assert self._server is not None
        try:
            await self._server.wait_closed()
        finally:
            await self.close()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def close(self) -> None:
        watchdog = self._watchdog_task
        self._watchdog_task = None
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)

        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        self._bound_port = None

    async def _idle_watchdog(self) -> None:
        while True:
            await asyncio.sleep(min(5, max(0.1, self._idle_timeout)))
            if self._active_inferences:
                continue
            if time.monotonic() - self._last_active <= self._idle_timeout:
                continue
            logging.info(
                "Idle timeout (%ss) reached; closing WebSocket server",
                self._idle_timeout,
            )
            server = self._server
            if server is not None:
                server.close()
                await server.wait_closed()
            return

    async def _handler(
        self,
        websocket: websockets.asyncio.server.ServerConnection,
    ) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                frame = await websocket.recv()
            except ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                return

            self._last_active = time.monotonic()
            if isinstance(frame, str):
                response = protocol.error_response(
                    message="Requests must be binary MessagePack frames",
                    code="invalid_frame",
                )
            else:
                try:
                    message = msgpack_numpy.unpackb(frame)
                except Exception as exc:
                    logging.warning("Failed to decode MessagePack request: %s", exc)
                    response = protocol.error_response(
                        message="Malformed MessagePack request",
                        code="invalid_message",
                    )
                else:
                    response = await self._route_message(message)

            try:
                encoded = packer.pack(response)
            except Exception:
                logging.exception("Failed to encode policy response")
                request_id = (
                    response.get("request_id", "default")
                    if isinstance(response, dict)
                    else "default"
                )
                message_type = (
                    response.get("type", "error")
                    if isinstance(response, dict)
                    else "error"
                )
                encoded = packer.pack(
                    protocol.error_response(
                        request_id=request_id,
                        message="Response serialization failed",
                        code="serialization_error",
                        message_type=message_type,
                    )
                )
            try:
                await websocket.send(encoded)
            except ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                return

    async def _route_message(self, message: Any) -> dict[str, Any]:
        request_id = (
            message.get("request_id", "default")
            if isinstance(message, dict)
            else "default"
        )
        requested_type = (
            message.get("type", protocol.INFER)
            if isinstance(message, dict)
            else "error"
        )
        try:
            message_type, request_id, payload = protocol.decode_request(message)
        except protocol.ProtocolError as exc:
            return protocol.error_response(
                request_id=request_id,
                message=str(exc),
                code="invalid_request",
                message_type=(
                    requested_type if isinstance(requested_type, str) else "error"
                ),
            )

        handler = self._handlers.get(message_type)
        if handler is None:
            return protocol.error_response(
                request_id=request_id,
                message=f"Unsupported message type {message_type!r}",
                code="unknown_message_type",
                message_type="unknown",
            )

        try:
            return await handler(message_type, request_id, payload)
        except Exception as exc:
            logging.exception(
                "Policy route failed (type=%s, request_id=%s)",
                message_type,
                request_id,
            )
            response_type = (
                "inference_result"
                if message_type in {protocol.INFER, protocol.PREDICT_ACTION}
                else message_type
            )
            return protocol.error_response(
                request_id=request_id,
                message=str(exc),
                code="handler_error",
                message_type=response_type,
            )

    async def _handle_ping(
        self,
        message_type: str,
        request_id: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del message_type, payload
        return protocol.success_response(
            request_id=request_id,
            message_type=protocol.PING,
        )

    async def _handle_infer(
        self,
        message_type: str,
        request_id: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del message_type
        async with self._inference_lock:
            self._active_inferences += 1
            try:
                output = await asyncio.to_thread(self._policy.predict_action, **payload)
            finally:
                self._active_inferences -= 1
                self._last_active = time.monotonic()
        if not isinstance(output, dict):
            raise TypeError("Policy predict_action must return a dict")
        return protocol.success_response(
            request_id=request_id,
            message_type="inference_result",
            data=output,
        )

    async def _handle_unsupported_capability(
        self,
        message_type: str,
        request_id: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del payload
        return protocol.error_response(
            request_id=request_id,
            message=(
                f"{message_type!r} is reserved for a future streaming backend "
                "and isn't supported by this server"
            ),
            code="capability_not_supported",
            message_type=message_type,
        )
