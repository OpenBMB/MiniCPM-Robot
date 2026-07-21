# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standard-library helpers shared by simulation launchers."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path


def resolve_executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if "/" in value:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"{label} is not executable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"{label} is not available on PATH: {value}")
    return resolved


def resolve_checkpoint(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Local checkpoint must be a directory: {candidate}")
        return str(candidate.resolve())
    if value.startswith(("/", "./", "../")):
        raise ValueError(f"Local checkpoint directory does not exist: {value}")
    return value


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def port_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def terminate_process_group(
    process: subprocess.Popen | None,
    *,
    timeout: float = 5,
) -> None:
    if process is None:
        return
    process_group = process.pid
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=timeout)
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is not None:
        time.sleep(0.05)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def prepend_pythonpath(env: dict[str, str], *paths: Path) -> dict[str, str]:
    result = dict(env)
    parts = [str(path) for path in paths]
    if result.get("PYTHONPATH"):
        parts.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = ":".join(parts)
    return result


def wait_for_policy_server(
    *,
    server: subprocess.Popen,
    minicpm_python: str,
    root: Path,
    host: str,
    port: int,
    checkpoint: str,
    embodiment_id: int,
    min_action_dim: int,
    request_id: str,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    env = prepend_pythonpath(os.environ, root)
    while time.monotonic() < deadline:
        status = server.poll()
        if status is not None:
            raise RuntimeError(
                f"Policy server exited before readiness with status {status}"
            )
        remaining = max(0.1, deadline - time.monotonic())
        try:
            probe = subprocess.run(
                [
                    minicpm_python,
                    "-m",
                    "evaluation.common.probe_server",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--checkpoint",
                    checkpoint,
                    "--embodiment-id",
                    str(embodiment_id),
                    "--min-action-dim",
                    str(min_action_dim),
                    "--request-id",
                    request_id,
                    "--timeout",
                    str(min(2.0, remaining)),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=min(5.0, remaining + 1),
                check=False,
            )
        except subprocess.TimeoutExpired:
            last_error = "readiness probe timed out"
        else:
            if probe.returncode == 0:
                return
            last_error = (probe.stderr or probe.stdout).strip()
            if probe.returncode == 3:
                raise RuntimeError(last_error or "Server metadata mismatch")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"Timed out waiting for {host}:{port}: {last_error}")
