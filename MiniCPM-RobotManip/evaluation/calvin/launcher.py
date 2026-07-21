# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Launch one MiniCPM server and the serial CALVIN evaluator."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from evaluation.common.launcher_utils import (
    git_revision,
    prepend_pythonpath,
    resolve_checkpoint,
    resolve_executable,
    terminate_process_group,
    wait_for_policy_server,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the serial CALVIN evaluation. Runtime paths and model "
            "settings are provided through environment variables; see README.md."
        )
    )
    return parser


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"Set required environment variable {name}")
    return value


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return default if value is None else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent

    try:
        minicpm_python = resolve_executable(
            require_env("MINICPM_PYTHON"),
            "MINICPM_PYTHON",
        )
        calvin_python = resolve_executable(
            require_env("CALVIN_PYTHON"),
            "CALVIN_PYTHON",
        )
        calvin_root = Path(require_env("CALVIN_ROOT")).expanduser().resolve()
        dataset_path = Path(
            require_env("CALVIN_DATASET_PATH")
        ).expanduser().resolve()
        checkpoint = resolve_checkpoint(
            os.environ.get("CHECKPOINT", "openbmb/MiniCPM-RobotManip")
        )
        embodiment_id = int(require_env("EMBODIMENT_ID"))
        port = env_int("PORT", 10093)
        num_sequences = env_int("NUM_SEQUENCES", 1000)
        resize_size = env_int("RESIZE_SIZE", 448)
        seed = env_int("SEED", 0)
        timeout = env_int("READY_TIMEOUT", 600)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if embodiment_id < 0:
        print("EMBODIMENT_ID must be non-negative.", file=sys.stderr)
        return 2
    if not 1 <= port <= 65535:
        print("PORT must be in [1, 65535].", file=sys.stderr)
        return 2
    if num_sequences < 1 or resize_size < 1 or timeout < 1:
        print(
            "NUM_SEQUENCES, RESIZE_SIZE, and READY_TIMEOUT must be positive.",
            file=sys.stderr,
        )
        return 2

    host = os.environ.get("HOST", "127.0.0.1")
    config_path = Path(
        os.environ.get(
            "CALVIN_CONFIG_PATH",
            calvin_root / "calvin_models" / "conf",
        )
    ).expanduser()
    sequences_path = Path(
        os.environ.get(
            "EVAL_SEQUENCES_PATH",
            script_dir / "eval_sequences.json",
        )
    ).expanduser()
    output_root = Path(
        os.environ.get(
            "OUTPUT_ROOT",
            root / "outputs" / "evaluation" / "calvin",
        )
    ).expanduser()
    run_id = os.environ.get(
        "RUN_ID",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    log_dir = Path(
        os.environ.get("EVAL_LOG_DIR", output_root / run_id)
    ).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "policy_server.log"
    evaluator_log = log_dir / "evaluator.log"

    manifest_fields = {
        "checkpoint": checkpoint,
        "default_embodiment_id": str(embodiment_id),
        "minicpm_revision": git_revision(root),
        "calvin_revision": git_revision(calvin_root),
        "calvin_dataset_path": str(dataset_path),
        "eval_sequences_path": str(sequences_path),
        "num_sequences": str(num_sequences),
        "host": host,
        "port": str(port),
    }
    with (log_dir / "run_manifest.tsv").open("w", encoding="utf-8") as file:
        for key, value in manifest_fields.items():
            file.write(f"{key}\t{value}\n")

    server: subprocess.Popen | None = None
    evaluator: subprocess.Popen | None = None
    interrupted = 0

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
                    "MINICPM_PYTHON": minicpm_python,
                    "CHECKPOINT": checkpoint,
                    "EMBODIMENT_ID": str(embodiment_id),
                    "HOST": host,
                    "PORT": str(port),
                },
                start_new_session=True,
            )
            wait_for_policy_server(
                server=server,
                minicpm_python=minicpm_python,
                root=root,
                host=host,
                port=port,
                checkpoint=checkpoint,
                embodiment_id=embodiment_id,
                min_action_dim=7,
                request_id="calvin-readiness",
                timeout=timeout,
            )
            if interrupted:
                raise RuntimeError("Evaluation interrupted")

            eval_command = [
                calvin_python,
                "-m",
                "evaluation.calvin.eval_calvin",
                "--args.host",
                host,
                "--args.port",
                str(port),
                "--args.resize-size",
                str(resize_size),
                "--args.calvin-root",
                str(calvin_root),
                "--args.dataset-path",
                str(dataset_path),
                "--args.calvin-config-path",
                str(config_path),
                "--args.eval-sequences-path",
                str(sequences_path),
                "--args.num-sequences",
                str(num_sequences),
                "--args.seed",
                str(seed),
                "--args.eval-log-dir",
                str(log_dir),
            ]
            annotation_cache = os.environ.get("LANG_ANNOTATION_CACHE")
            if annotation_cache:
                eval_command.extend(
                    ["--args.lang-annotation-cache", annotation_cache]
                )
            if enabled("DEBUG"):
                eval_command.append("--args.debug")
            if enabled("RESET"):
                eval_command.append("--args.reset")
            if enabled("DIVERSE_INST"):
                eval_command.append("--args.diverse-inst")

            eval_env = prepend_pythonpath(
                os.environ,
                root,
                calvin_root,
                calvin_root / "calvin_models",
                calvin_root / "calvin_env",
            )
            with evaluator_log.open("w", encoding="utf-8") as eval_stream:
                evaluator = subprocess.Popen(
                    eval_command,
                    stdout=eval_stream,
                    stderr=subprocess.STDOUT,
                    env=eval_env,
                    start_new_session=True,
                )
                if interrupted:
                    raise RuntimeError("Evaluation interrupted")
                status = evaluator.wait()

            if status:
                print(
                    f"CALVIN evaluator failed with status {status}; "
                    f"see {evaluator_log}",
                    file=sys.stderr,
                )
                return status
            if server.poll() is not None:
                print(
                    "MiniCPM server exited before CALVIN completed.",
                    file=sys.stderr,
                )
                return 1
            print(f"CALVIN evaluation completed; outputs: {log_dir}")
            return 0
    except Exception as exc:
        if interrupted:
            return 128 + interrupted
        print(f"CALVIN launcher failed: {exc}; see {server_log}", file=sys.stderr)
        return 1
    finally:
        terminate_process_group(evaluator)
        terminate_process_group(server)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
