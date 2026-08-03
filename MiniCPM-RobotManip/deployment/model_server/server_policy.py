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

"""Launch a starVLA-compatible MiniCPM-RobotManip WebSocket server."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from deployment.model_server.policy_wrapper import MiniCPMPolicyWrapper
from deployment.model_server.protocol import build_server_metadata
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _load_runner(checkpoint: str, device: str | None) -> Any:
    # Keep model imports and checkpoint loading out of module import so --help and
    # protocol-only tests don't require a GPU or download model files.
    from vla_infer import MiniCPMVLAInference

    return MiniCPMVLAInference(
        checkpoint_path=checkpoint,
        device=device,
    )


def main(args: argparse.Namespace) -> None:
    logging.info("Loading MiniCPM-RobotManip from %s", args.checkpoint)
    runner = _load_runner(args.checkpoint, args.device)
    policy = MiniCPMPolicyWrapper(
        runner=runner,
        default_embodiment_id=args.default_embodiment_id,
    )
    metadata = build_server_metadata(
        policy.metadata,
        checkpoint=args.checkpoint,
    )
    logging.warning(
        "Serving execution-ready MiniCPM actions without normalization or "
        "un-normalization. Verify camera order and embodiment_id for the target robot."
    )
    logging.info("Server metadata: %s", metadata)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
        max_message_bytes=args.max_message_bytes,
    )
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="openbmb/MiniCPM-RobotManip",
        help="Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device; default is CUDA when available, otherwise CPU",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument(
        "--default-embodiment-id",
        type=int,
        default=0,
        help="Used when a starVLA evaluator doesn't send embodiment_id",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=-1,
        help="Server idle timeout in seconds; -1 disables automatic shutdown",
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="Maximum incoming WebSocket frame size",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    main(parsed_args)
