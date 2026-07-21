# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Run one MiniCPM server and one LIBERO suite evaluation."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from evaluation.common.launcher_utils import (
    git_revision,
    port_available,
    resolve_checkpoint,
    resolve_executable,
    terminate_process_group,
    wait_for_policy_server,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("task_suite")
    parser.add_argument("gpu_id")
    parser.add_argument("port", type=int)
    return parser


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "unnamed"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("port must be in [1, 65535]", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    libero_value = os.environ.get("LIBERO_HOME", "")
    if not libero_value:
        print("LIBERO_HOME is required.", file=sys.stderr)
        return 2
    libero_home = Path(libero_value).expanduser()
    if not libero_home.is_dir():
        print(f"LIBERO_HOME doesn't exist: {libero_home}", file=sys.stderr)
        return 2

    embodiment_value = os.environ.get("EMBODIMENT_ID", "")
    try:
        embodiment_id = int(embodiment_value)
    except ValueError:
        print("EMBODIMENT_ID is required and must be an integer.", file=sys.stderr)
        return 2
    if embodiment_id < 0:
        print("EMBODIMENT_ID must be non-negative.", file=sys.stderr)
        return 2

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        minicpm_python = resolve_executable(
            os.environ.get("MINICPM_PYTHON", "python"),
            "MINICPM_PYTHON",
        )
        libero_python = resolve_executable(
            os.environ.get("LIBERO_PYTHON", "python"),
            "LIBERO_PYTHON",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    host = os.environ.get("HOST", "127.0.0.1")
    timeout = int(os.environ.get("SERVER_TIMEOUT", "300"))
    if timeout < 1:
        print("SERVER_TIMEOUT must be positive.", file=sys.stderr)
        return 2
    if not port_available(host, args.port):
        print(f"Port {host}:{args.port} is already in use.", file=sys.stderr)
        return 1

    output_root = Path(
        os.environ.get(
            "OUTPUT_ROOT",
            root / "outputs" / "evaluation" / "libero",
        )
    ).expanduser()
    run_id = os.environ.get("RUN_ID", "manual")
    job_id = os.environ.get("JOB_ID", "")
    checkpoint_label = safe_label(checkpoint)
    prefix = f"job-{job_id}-" if job_id else ""
    job_root = output_root / run_id / f"{prefix}{checkpoint_label}"
    job_dir = job_root / args.task_suite
    job_dir.mkdir(parents=True, exist_ok=True)
    server_log = job_dir / "server.log"
    eval_log = job_dir / "eval.log"
    manifest = job_dir / "manifest.tsv"

    fields = {
        "source": "starVLA@631aae02afe6d95876e923ff518e8ff2ab9a2f88",
        "checkpoint": checkpoint,
        "minicpm_revision": git_revision(root),
        "libero_revision": git_revision(libero_home),
        "task_suite": args.task_suite,
        "gpu": args.gpu_id,
        "egl_device": os.environ.get("EGL_DEVICE_ID", "0"),
        "host": host,
        "port": str(args.port),
        "embodiment_id": str(embodiment_id),
        "server_python": minicpm_python,
        "libero_python": libero_python,
        "server_log": str(server_log),
        "eval_log": str(eval_log),
    }
    with manifest.open("w", encoding="utf-8") as file:
        for key, value in fields.items():
            file.write(f"{key}\t{value}\n")

    server: subprocess.Popen | None = None
    evaluator: subprocess.Popen | None = None
    interrupted = 0
    final_status = 1

    def handle_signal(signum, frame) -> None:
        del frame
        nonlocal interrupted
        interrupted = signum
        terminate_process_group(evaluator)
        terminate_process_group(server)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, handle_signal)

    try:
        with server_log.open("w", encoding="utf-8") as server_stream:
            server = subprocess.Popen(
                ["bash", str(script_dir / "run_policy_server.sh")],
                stdout=server_stream,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "MINICPM_ROOT": str(root),
                    "MINICPM_PYTHON": minicpm_python,
                    "CHECKPOINT": checkpoint,
                    "EMBODIMENT_ID": str(embodiment_id),
                    "HOST": host,
                    "PORT": str(args.port),
                    "GPU_ID": args.gpu_id,
                },
                start_new_session=True,
            )
            wait_for_policy_server(
                server=server,
                minicpm_python=minicpm_python,
                root=root,
                host=host,
                port=args.port,
                checkpoint=checkpoint,
                embodiment_id=embodiment_id,
                min_action_dim=7,
                request_id="libero-readiness",
                timeout=timeout,
            )
            if interrupted:
                raise RuntimeError("Evaluation interrupted")

            with eval_log.open("w", encoding="utf-8") as eval_stream:
                evaluator = subprocess.Popen(
                    ["bash", str(script_dir / "eval_libero.sh")],
                    stdout=eval_stream,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "MINICPM_ROOT": str(root),
                        "LIBERO_HOME": str(libero_home),
                        "LIBERO_PYTHON": libero_python,
                        "OUTPUT_ROOT": str(job_root),
                        "HOST": host,
                        "PORT": str(args.port),
                        "TASK_SUITE_NAME": args.task_suite,
                        "CUDA_VISIBLE_DEVICES": args.gpu_id,
                    },
                    start_new_session=True,
                )
                if interrupted:
                    raise RuntimeError("Evaluation interrupted")
                eval_status = evaluator.wait()

            server_alive = server.poll() is None
            if eval_status:
                final_status = eval_status
                print(
                    f"LIBERO evaluation failed with status {eval_status}; "
                    f"see {eval_log}",
                    file=sys.stderr,
                )
                return eval_status
            if not server_alive:
                final_status = 1
                print(
                    "MiniCPM server exited before evaluation completed.",
                    file=sys.stderr,
                )
                return 1
            print(f"Evaluation completed. Outputs: {job_dir}")
            final_status = 0
            return 0
    except Exception as exc:
        if interrupted:
            final_status = 128 + interrupted
            return final_status
        print(f"LIBERO worker failed: {exc}; see {server_log}", file=sys.stderr)
        final_status = 1
        return 1
    finally:
        terminate_process_group(evaluator)
        terminate_process_group(server)
        with manifest.open("a", encoding="utf-8") as file:
            file.write(f"exit_status\t{final_status}\n")
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
