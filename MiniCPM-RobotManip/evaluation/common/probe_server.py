# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate MiniCPM policy metadata and ping readiness."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from deployment.model_server.tools.websocket_policy_client import (
    WebsocketClientPolicy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment-id", type=int, required=True)
    parser.add_argument("--min-action-dim", type=int, required=True)
    parser.add_argument("--request-id", default="evaluation-readiness")
    parser.add_argument("--timeout", type=float, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = WebsocketClientPolicy(
            args.host,
            args.port,
            open_timeout=args.timeout,
            response_timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Policy server isn't ready: {exc}", file=sys.stderr)
        return 2

    try:
        metadata = client.get_server_metadata()
        checks = {
            "server": metadata.get("server") == "minicpm_robot_manip",
            "checkpoint": metadata.get("ckpt_path") == args.checkpoint,
            "embodiment": (
                metadata.get("default_embodiment_id") == args.embodiment_id
            ),
            "normalization": metadata.get("action_normalization") == "none",
            "execution_ready": (
                metadata.get("actions_ready_for_execution") is True
            ),
        }
        action_dim = metadata.get("action_dim")
        chunk_size = metadata.get("action_chunk_size")
        checks["action_dim"] = (
            isinstance(action_dim, int)
            and not isinstance(action_dim, bool)
            and action_dim >= args.min_action_dim
        )
        checks["action_chunk_size"] = (
            isinstance(chunk_size, int)
            and not isinstance(chunk_size, bool)
            and chunk_size > 0
        )
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            print(
                f"Policy metadata mismatch ({', '.join(failed)}): {metadata!r}",
                file=sys.stderr,
            )
            return 3

        response = client.ping(request_id=args.request_id)
        if response.get("ok") is not True or response.get("type") != "ping":
            print(f"Policy ping failed: {response!r}", file=sys.stderr)
            return 3
        return 0
    except Exception as exc:
        print(f"Policy readiness validation failed: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
