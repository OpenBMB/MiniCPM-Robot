#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
import os
from pathlib import Path
import sys
import time


CLIENT_TOKEN = "client/http_minicpm_robot_track_client.py"
DEFAULT_CYCLONEDDS_HOME = "/home/unitree/cyclonedds/install"


def camera_client_running():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if CLIENT_TOKEN in command:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Check the Go2 VideoClient camera")
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", default="/tmp/go2-camera-sample.jpg")
    args = parser.parse_args()

    if camera_client_running():
        raise SystemExit("A camera client is already running. Stop it before running the standalone camera check.")

    cyclonedds_home = str(Path(os.environ.get("CYCLONEDDS_HOME", DEFAULT_CYCLONEDDS_HOME)).expanduser())
    required = [
        str(Path(cyclonedds_home) / "lib"),
        "/opt/ros/humble/lib",
        "/opt/ros/humble/lib/aarch64-linux-gnu",
    ]
    if os.environ.get("_GO2_CAMERA_ENV_READY") != "1":
        env = os.environ.copy()
        env["CYCLONEDDS_HOME"] = cyclonedds_home
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(required + ([current] if current else []))
        env["_GO2_CAMERA_ENV_READY"] = "1"
        os.execve("/usr/bin/python3", ["/usr/bin/python3", *sys.argv], env)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    ChannelFactoryInitialize(0, args.interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    start = time.perf_counter()
    received = 0
    last = None
    for _ in range(max(1, args.frames)):
        code, data = client.GetImageSample()
        if code != 0:
            print(f"frame error code={code}")
            continue
        last = bytes(data)
        received += 1
    elapsed = time.perf_counter() - start
    if last is None:
        raise SystemExit("No camera frame received")
    Path(args.output).write_bytes(last)
    print(
        f"camera OK: interface={args.interface} frames={received}/{args.frames} "
        f"fps={received / max(elapsed, 1e-6):.2f} last_jpeg_bytes={len(last)} output={args.output}"
    )


if __name__ == "__main__":
    main()
