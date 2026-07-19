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
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

import yaml


ROOT = Path(__file__).resolve().parent
REALWORLD = ROOT / "realworld"
LOG_DIR = ROOT / "logs"
SERVER_LOG = LOG_DIR / "server-maxn-fast.log"
REALSENSE_LOG = LOG_DIR / "realsense-d435i-rgb.log"
SERVER_TOKEN = "sample/http_minicpm_robot_track_server.py"
CLIENT_TOKEN = "client/http_minicpm_robot_track_client.py"
REALSENSE_TOKEN = "realsense2_camera_node"
HEALTH_URL = "http://127.0.0.1:5801/health"
STATE_URL = "http://127.0.0.1:5801/api/state"
CAMERA_CHECK = ROOT / "check_go2_camera.py"
D435I_RGB_TOPIC = "/camera/color/image_raw"
D435I_CAMERA_INFO_TOPIC = "/camera/color/camera_info"
USB_C_ROLE = Path("/sys/class/usb_role/usb2-0-role-switch/role")
SERVER_RESPONSE_MODE = "control"
DEFAULT_RUN_CONFIG_PATH = ROOT / "go2_runtime.yaml"
CHECKPOINT_ROOT = ROOT / "minicpm_robot_track/checkpoints"
DEFAULT_CYCLONEDDS_HOME = "/home/unitree/cyclonedds/install"
WAYPOINT_STRATEGIES = (
    "first",
    "two-step",
    "dx4-dw1",
)

RUN_CONFIG_DEFAULTS = {
    "mode": "dry-run",
    "instruction": "Follow the person ahead",
    "camera_source": "go2",
    "waypoint_strategy": "first",
    "skip_jetson_clocks": False,
    "model_checkpoint": "MiniCPM-RobotTrack",
    "video_network_interface": "enP8p1s0",
    "video_client_timeout": 3.0,
    "jpeg_quality": 60,
    "jpeg_encoder": "opencv",
    "rgb_decode": "fast",
    "upload_width": 384,
    "model_input_mode": "center_crop_height",
    "model_crop_size": 384,
    "native_preview": True,
    "camera_preview_fps": 10.0,
    "max_wait_new_frame_ms": 35.0,
    "fresh_frame_age_threshold_ms": 0.0,
    "vx_positive_scale": 1.0,
    "vx_negative_scale": 1.0,
    "wz_scale": 1.0,
    "vx_positive_deadband": 0.0,
    "vx_negative_deadband": 0.0,
    "deadband_wz": 0.0,
    "max_vx": 0.15,
    "max_wz": 0.30,
    "hysteresis_vx": 0.0,
    "ema_alpha": 1.0,
    "yaw_boost": 1.0,
    "yaw_boost_threshold": 0.10,
    "yaw_boost_max": 0.30,
    "latency_print_every": 10,
    "record_videos": False,
    "record_video_fps": 15,
    "record_video_segment_seconds": 10,
}

RUN_CONFIG_PATHS = {
    ("runtime", "mode"): "mode",
    ("runtime", "instruction"): "instruction",
    ("runtime", "camera_source"): "camera_source",
    ("runtime", "waypoint_strategy"): "waypoint_strategy",
    ("runtime", "skip_jetson_clocks"): "skip_jetson_clocks",
    ("model", "checkpoint"): "model_checkpoint",
    ("camera", "video_network_interface"): "video_network_interface",
    ("camera", "video_client_timeout"): "video_client_timeout",
    ("camera", "jpeg_quality"): "jpeg_quality",
    ("camera", "jpeg_encoder"): "jpeg_encoder",
    ("camera", "rgb_decode"): "rgb_decode",
    ("camera", "upload_width"): "upload_width",
    ("camera", "model_input_mode"): "model_input_mode",
    ("camera", "model_crop_size"): "model_crop_size",
    ("camera", "native_preview"): "native_preview",
    ("camera", "preview_fps"): "camera_preview_fps",
    ("camera", "max_wait_new_frame_ms"): "max_wait_new_frame_ms",
    ("camera", "fresh_frame_age_threshold_ms"): "fresh_frame_age_threshold_ms",
    ("velocity", "vx", "positive_scale"): "vx_positive_scale",
    ("velocity", "vx", "negative_scale"): "vx_negative_scale",
    ("velocity", "vx", "positive_deadband"): "vx_positive_deadband",
    ("velocity", "vx", "negative_deadband"): "vx_negative_deadband",
    ("velocity", "vx", "max_abs"): "max_vx",
    ("velocity", "vx", "sign_hysteresis"): "hysteresis_vx",
    ("velocity", "wz", "scale"): "wz_scale",
    ("velocity", "wz", "deadband"): "deadband_wz",
    ("velocity", "wz", "max_abs"): "max_wz",
    ("velocity", "wz", "boost"): "yaw_boost",
    ("velocity", "wz", "boost_threshold"): "yaw_boost_threshold",
    ("velocity", "wz", "boost_max"): "yaw_boost_max",
    ("velocity", "ema_alpha"): "ema_alpha",
    ("logging", "latency_print_every"): "latency_print_every",
    ("logging", "record_videos"): "record_videos",
    ("logging", "video_fps"): "record_video_fps",
    ("logging", "video_segment_seconds"): "record_video_segment_seconds",
}


def _flatten_config(mapping, prefix=()):
    for key, value in mapping.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from _flatten_config(value, path)
        else:
            yield path, value


def load_run_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise RuntimeError(f"Runtime config does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"Runtime config must contain a YAML mapping: {config_path}")

    config = dict(RUN_CONFIG_DEFAULTS)
    unknown = []
    for yaml_path, value in _flatten_config(raw):
        key = RUN_CONFIG_PATHS.get(yaml_path)
        if key is None:
            unknown.append(".".join(yaml_path))
        else:
            config[key] = value
    if unknown:
        raise RuntimeError("Unknown runtime config field(s): " + ", ".join(sorted(unknown)))
    return config


def _number(config, key, integer=False):
    value = config[key]
    if isinstance(value, bool):
        raise RuntimeError(f"Config {key} must be numeric")
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Config {key} must be numeric, got {value!r}") from exc
    if integer and result != value:
        raise RuntimeError(f"Config {key} must be an integer, got {value!r}")
    if not integer and not math.isfinite(result):
        raise RuntimeError(f"Config {key} must be finite")
    config[key] = result


def resolve_model_checkpoint_path(value):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Config model_checkpoint must be a non-empty string")
    checkpoint = Path(value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = CHECKPOINT_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    required_snapshot_files = (
        checkpoint / "config.json",
        checkpoint / "configuration_robottrack.py",
        checkpoint / "configuration_minicpm.py",
        checkpoint / "modeling_robottrack.py",
        checkpoint / "modeling_minicpm.py",
        checkpoint / "tokenizer_config.json",
    )
    weight_files = (
        checkpoint / "model.safetensors",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "pytorch_model.bin",
        checkpoint / "pytorch_model.bin.index.json",
    )
    missing = [path.name for path in required_snapshot_files if not path.is_file()]
    has_weights = any(path.is_file() for path in weight_files)
    if not has_weights:
        missing.append("model weights")
    if missing:
        raise RuntimeError(
            f"Model checkpoint is not a loadable Hugging Face directory: {checkpoint}. "
            "Expected the complete MiniCPM-RobotTrack snapshot with config, custom "
            f"model code, tokenizer files, and model weights; missing={missing}."
        )
    return checkpoint


def validate_run_config(config):
    for key in (
        "video_client_timeout", "camera_preview_fps", "max_wait_new_frame_ms",
        "fresh_frame_age_threshold_ms", "vx_positive_scale", "vx_negative_scale",
        "wz_scale", "vx_positive_deadband", "vx_negative_deadband", "deadband_wz",
        "max_vx", "max_wz", "hysteresis_vx", "ema_alpha", "yaw_boost",
        "yaw_boost_threshold", "yaw_boost_max",
    ):
        _number(config, key)
    for key in (
        "jpeg_quality", "upload_width", "model_crop_size", "latency_print_every",
        "record_video_fps", "record_video_segment_seconds",
    ):
        _number(config, key, integer=True)

    for key in ("mode", "instruction", "camera_source", "waypoint_strategy", "model_checkpoint",
                "video_network_interface", "jpeg_encoder", "rgb_decode", "model_input_mode"):
        if not isinstance(config[key], str):
            raise RuntimeError(f"Config {key} must be a string")
    if not isinstance(config["skip_jetson_clocks"], bool):
        raise RuntimeError("Config skip_jetson_clocks must be true or false")
    if not isinstance(config["native_preview"], bool):
        raise RuntimeError("Config native_preview must be true or false")
    if not isinstance(config["record_videos"], bool):
        raise RuntimeError("Config record_videos must be true or false")
    if config["mode"] not in ("dry-run", "live"):
        raise RuntimeError("Config mode must be dry-run or live")
    if config["camera_source"] not in ("go2", "d435i"):
        raise RuntimeError("Config camera_source must be go2 or d435i")
    if config["waypoint_strategy"] not in WAYPOINT_STRATEGIES:
        raise RuntimeError(f"Config waypoint_strategy must be one of {WAYPOINT_STRATEGIES}")
    if config["jpeg_encoder"] not in ("opencv", "pil"):
        raise RuntimeError("Config jpeg_encoder must be opencv or pil")
    if config["rgb_decode"] not in ("fast", "cv_bridge"):
        raise RuntimeError("Config rgb_decode must be fast or cv_bridge")
    if config["model_input_mode"] not in ("aspect_resize", "center_crop_height", "center_crop_720p"):
        raise RuntimeError("Invalid model_input_mode in runtime config")
    if not config["instruction"].strip():
        raise RuntimeError("Config instruction must not be empty")
    if not config["model_checkpoint"].strip():
        raise RuntimeError("Config model_checkpoint must not be empty")
    if not config["video_network_interface"].strip():
        raise RuntimeError("Config video_network_interface must not be empty")
    if not 1 <= config["jpeg_quality"] <= 100:
        raise RuntimeError("Config jpeg_quality must be in [1, 100]")
    if config["upload_width"] <= 0 or config["model_crop_size"] <= 0:
        raise RuntimeError("Config upload_width and model_crop_size must be positive")
    for key in (
        "video_client_timeout", "camera_preview_fps", "max_wait_new_frame_ms",
        "fresh_frame_age_threshold_ms", "vx_positive_scale", "vx_negative_scale",
        "wz_scale", "vx_positive_deadband", "vx_negative_deadband", "deadband_wz",
        "hysteresis_vx", "yaw_boost", "yaw_boost_threshold", "yaw_boost_max",
    ):
        if config[key] < 0.0:
            raise RuntimeError(f"Config {key} must be non-negative")
    if not 0.0 < config["max_vx"] <= 1.2:
        raise RuntimeError("Config max_vx must be in (0, 1.2]")
    if not 0.0 < config["max_wz"] <= 1.5:
        raise RuntimeError("Config max_wz must be in (0, 1.5]")
    if config["vx_positive_deadband"] > config["max_vx"] or config["vx_negative_deadband"] > config["max_vx"]:
        raise RuntimeError("Directional vx deadbands must not exceed max_vx")
    if config["deadband_wz"] > config["max_wz"]:
        raise RuntimeError("Config deadband_wz must not exceed max_wz")
    if not 0.0 < config["ema_alpha"] <= 1.0:
        raise RuntimeError("Config ema_alpha must be in (0, 1]")
    if config["latency_print_every"] < 0:
        raise RuntimeError("Config latency_print_every must be non-negative")
    if not 1 <= config["record_video_fps"] <= 30:
        raise RuntimeError("Config record_video_fps must be in [1, 30]")
    if not 2 <= config["record_video_segment_seconds"] <= 300:
        raise RuntimeError("Config record_video_segment_seconds must be in [2, 300]")
    return config


def resolve_run_config(args):
    config = load_run_config(args.config)
    direct_overrides = {
        "mode": "mode",
        "instruction": "instruction",
        "camera_source": "camera_source",
        "waypoint_strategy": "waypoint_strategy",
        "skip_jetson_clocks": "skip_jetson_clocks",
        "model_checkpoint": "model_checkpoint",
        "video_network_interface": "video_network_interface",
        "video_client_timeout": "video_client_timeout",
        "jpeg_quality": "jpeg_quality",
        "jpeg_encoder": "jpeg_encoder",
        "rgb_decode": "rgb_decode",
        "upload_width": "upload_width",
        "model_input_mode": "model_input_mode",
        "model_crop_size": "model_crop_size",
        "native_preview": "native_preview",
        "camera_preview_fps": "camera_preview_fps",
        "max_wait_new_frame_ms": "max_wait_new_frame_ms",
        "fresh_frame_age_threshold_ms": "fresh_frame_age_threshold_ms",
        "wz_scale": "wz_scale",
        "max_vx": "max_vx",
        "max_wz": "max_wz",
        "deadband_wz": "deadband_wz",
        "hysteresis_vx": "hysteresis_vx",
        "ema_alpha": "ema_alpha",
        "yaw_boost": "yaw_boost",
        "yaw_boost_threshold": "yaw_boost_threshold",
        "yaw_boost_max": "yaw_boost_max",
        "latency_print_every": "latency_print_every",
        "record_videos": "record_videos",
        "record_video_fps": "record_video_fps",
        "record_video_segment_seconds": "record_video_segment_seconds",
    }
    for argument, key in direct_overrides.items():
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = value

    # Backward-compatible global shortcuts set both directions. Explicit
    # directional CLI options take precedence when both are supplied.
    if getattr(args, "vx_scale", None) is not None:
        config["vx_positive_scale"] = args.vx_scale
        config["vx_negative_scale"] = args.vx_scale
    if getattr(args, "deadband_vx", None) is not None:
        config["vx_positive_deadband"] = args.deadband_vx
        config["vx_negative_deadband"] = args.deadband_vx
    for argument, key in (
        ("vx_positive_scale", "vx_positive_scale"),
        ("vx_negative_scale", "vx_negative_scale"),
        ("vx_positive_deadband", "vx_positive_deadband"),
        ("vx_negative_deadband", "vx_negative_deadband"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = value
    return validate_run_config(config)


def local_json(url, timeout=2.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def matching_pids(token):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if token in cmdline and "go2_runtime.py" not in cmdline:
            matches.append((int(entry.name), cmdline.strip()))
    return matches


def wait_for_process_exit(token, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not matching_pids(token):
            return True
        time.sleep(0.1)
    return not matching_pids(token)


def signal_processes(token, sig):
    for pid, _cmdline in matching_pids(token):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop_control(send_stop=True, network_interface="enP8p1s0"):
    signal_processes(CLIENT_TOKEN, signal.SIGINT)
    if not wait_for_process_exit(CLIENT_TOKEN, 2.0):
        signal_processes(CLIENT_TOKEN, signal.SIGTERM)
        wait_for_process_exit(CLIENT_TOKEN, 1.0)
    if send_stop:
        emergency_stop(network_interface)


def stop_server():
    signal_processes(SERVER_TOKEN, signal.SIGTERM)
    wait_for_process_exit(SERVER_TOKEN, 2.0)


def unitree_env():
    env = os.environ.copy()
    cyclonedds_home = os.path.expanduser(
        env.get("CYCLONEDDS_HOME", DEFAULT_CYCLONEDDS_HOME)
    ).replace("\\", "/")
    env["CYCLONEDDS_HOME"] = cyclonedds_home
    required = [
        f"{cyclonedds_home.rstrip('/')}/lib",
        "/opt/ros/humble/lib",
        "/opt/ros/humble/lib/aarch64-linux-gnu",
    ]
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(required + ([current] if current else []))
    return env


def ros_env():
    command = "source /opt/ros/humble/setup.bash >/dev/null 2>&1 && env -0"
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True)
    env = {}
    for item in result.stdout.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode()] = value.decode()
    cyclonedds_home = os.path.expanduser(
        env.get("CYCLONEDDS_HOME", DEFAULT_CYCLONEDDS_HOME)
    ).replace("\\", "/")
    env["CYCLONEDDS_HOME"] = cyclonedds_home
    # VS Code terminals can retain DDS settings from earlier experiments.  The
    # Go2 client uses Unitree CycloneDDS directly, while ROS should stay on Fast
    # DDS.  Pin the known-good split so client startup is reproducible.
    env.pop("CYCLONEDDS_URI", None)
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    env["ROS_DOMAIN_ID"] = "0"
    required = [
        f"{cyclonedds_home.rstrip('/')}/lib",
        "/opt/ros/humble/lib",
        "/opt/ros/humble/lib/aarch64-linux-gnu",
    ]
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(required + ([current] if current else []))
    return env


def emergency_stop(network_interface="enP8p1s0"):
    code = """
import os
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
ChannelFactoryInitialize(0, os.environ['GO2_NETWORK_INTERFACE'])
client = SportClient()
client.SetTimeout(3.0)
client.Init()
for _ in range(3):
    client.Move(0.0, 0.0, 0.0)
    time.sleep(0.05)
result = client.StopMove()
print(f'StopMove sent, code={result}', flush=True)
"""
    try:
        env = unitree_env()
        env["GO2_NETWORK_INTERFACE"] = network_interface
        subprocess.run(["/usr/bin/python3", "-c", code], env=env, check=True, timeout=10)
    except Exception as exc:
        print(f"WARNING: StopMove failed: {exc}", file=sys.stderr)


def go2_camera_preflight(network_interface="enP8p1s0"):
    if not CAMERA_CHECK.exists():
        raise RuntimeError(f"Camera check script is missing: {CAMERA_CHECK}")
    print("Checking Go2 camera and DDS interface before starting the client.")
    command = [
        "/usr/bin/python3", str(CAMERA_CHECK),
        "--interface", network_interface,
        "--frames", "2",
        "--output", "/tmp/go2-camera-preflight.jpg",
    ]
    result = subprocess.run(command, cwd=ROOT, env=unitree_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Go2 camera preflight failed. Check {network_interface} and run: "
            f"./check_go2_camera.py --interface {network_interface} --frames 30"
        )
    # Let the short-lived CycloneDDS participant release its sockets fully.
    time.sleep(0.75)


def d435i_device_details():
    enumerator = Path("/opt/ros/humble/bin/rs-enumerate-devices")
    if not enumerator.exists():
        raise RuntimeError(
            "RealSense tools are missing. Install ros-humble-realsense2-camera first."
        )

    def enumerate_once():
        return subprocess.run(
            [str(enumerator), "--compact"],
            env=ros_env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    result = enumerate_once()
    if result.returncode != 0 and USB_C_ROLE.exists():
        role = USB_C_ROLE.read_text().strip()
        print(
            f"D435i is not enumerated while the Orin USB-C role reports {role}; "
            "rebinding none -> host so the host controller enables VBUS. "
            "This disables the port's USB gadget/RNDIS link."
        )
        try:
            subprocess.run(
                ["sudo", "systemctl", "stop", "force-usb-device-mode.service"],
                check=True,
            )
            subprocess.run(
                ["sudo", "systemctl", "stop", "nv-l4t-usb-device-mode.service"],
                check=True,
            )
            for next_role in ("none", "host"):
                subprocess.run(
                    ["sudo", "tee", str(USB_C_ROLE)],
                    input=next_role + "\n",
                    text=True,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                time.sleep(0.25)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to rebind the Orin USB-C port to host mode with sudo"
            ) from exc
        time.sleep(2.0)
        result = enumerate_once()

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        role = USB_C_ROLE.read_text().strip() if USB_C_ROLE.exists() else "unavailable"
        raise RuntimeError(
            "D435i is not visible on the Orin NX USB bus "
            f"(USB-C role={role}). Unplug/replug it after selecting host mode and run `lsusb`; "
            "if a USB-C-to-USB-C cable still does not enumerate, use the Orin USB 3 Type-A "
            "host port with a USB-A-to-USB-C data cable. RealSense said: "
            f"{output or 'no device detected'}"
        )
    return output


def ros_topic_names():
    result = subprocess.run(
        ["ros2", "topic", "list", "--no-daemon"],
        env=ros_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def wait_for_rgb_frame(topic, timeout=10.0):
    try:
        result = subprocess.run(
            [
                "ros2", "topic", "echo", topic, "sensor_msgs/msg/Image",
                "--no-daemon", "--once", "--field", "width",
            ],
            env=ros_env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and any(char.isdigit() for char in result.stdout)


def stop_realsense(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass


def create_run_log_bundle(mode, camera_source, waypoint_strategy, config):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = LOG_DIR / "runs" / f"{timestamp}-{os.getpid()}-{mode}-{camera_source}-{waypoint_strategy}"
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "argv": sys.argv,
        "config": config,
        "logs": {
            "client": str(session_dir / "client.log"),
            "diagnostics": str(session_dir / "diagnostics.jsonl"),
            "tegrastats": str(session_dir / "tegrastats.log"),
            "server": str(SERVER_LOG),
            "realsense": str(REALSENSE_LOG),
        },
    }
    manifest_path = session_dir / "session.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    latest_path = LOG_DIR / "latest-run.txt"
    with latest_path.open("w", encoding="utf-8") as latest_file:
        latest_file.write(str(session_dir) + "\n")
        latest_file.flush()
        os.fsync(latest_file.fileno())
    return session_dir


def _write_tegrastats_output(process, log_path):
    last_sync = time.monotonic()
    with Path(log_path).open("w", buffering=1) as log_handle:
        log_handle.write(f"# started_at={datetime.now(timezone.utc).isoformat()} interval_ms=1000\n")
        log_handle.flush()
        os.fsync(log_handle.fileno())
        for line in process.stdout:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            log_handle.write(f"{timestamp} {line}")
            now = time.monotonic()
            if now - last_sync >= 1.0:
                log_handle.flush()
                os.fsync(log_handle.fileno())
                last_sync = now
        log_handle.flush()
        os.fsync(log_handle.fileno())


def start_tegrastats_log(log_path):
    command = ["/usr/bin/tegrastats", "--interval", "1000"]
    if Path("/usr/bin/stdbuf").is_file():
        command = ["/usr/bin/stdbuf", "-oL", *command]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    writer = threading.Thread(
        target=_write_tegrastats_output,
        args=(process, log_path),
        name="tegrastats-log-writer",
        daemon=True,
    )
    writer.start()
    return process, writer


def stop_background_process(process_handle):
    process, writer = process_handle if isinstance(process_handle, tuple) else (process_handle, None)
    if process is None or process.poll() is not None:
        if writer is not None:
            writer.join(timeout=2.0)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
    if writer is not None:
        writer.join(timeout=2.0)


def run_client_with_tee(command, cwd, env, log_path):
    last_sync = time.monotonic()
    with Path(log_path).open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"# started_at={datetime.now(timezone.utc).isoformat()}\n")
        log_file.write("# command=" + " ".join(command) + "\n")
        log_file.flush()
        os.fsync(log_file.fileno())
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                now = time.monotonic()
                if now - last_sync >= 1.0:
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    last_sync = now
            return process.wait()
        finally:
            log_file.flush()
            os.fsync(log_file.fileno())


def start_realsense_rgb():
    details = d435i_device_details()
    print(f"D435i detected by librealsense: {details.splitlines()[0]}")

    if D435I_RGB_TOPIC in ros_topic_names():
        if wait_for_rgb_frame(D435I_RGB_TOPIC):
            print(f"Reusing existing RealSense RGB topic {D435I_RGB_TOPIC}.")
            return None
        raise RuntimeError(
            f"RealSense topic {D435I_RGB_TOPIC} exists but did not deliver an RGB frame"
        )
    if matching_pids(REALSENSE_TOKEN):
        raise RuntimeError(
            "A RealSense node is already running without the expected RGB topic. "
            f"Expected {D435I_RGB_TOPIC}; stop the stale node before retrying."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = REALSENSE_LOG.open("w")
    command = [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        "camera_namespace:=/",
        "camera_name:=camera",
        "enable_color:=true",
        "rgb_camera.color_profile:=640,480,30",
        "enable_depth:=false",
        "enable_infra:=false",
        "enable_infra1:=false",
        "enable_infra2:=false",
        "enable_gyro:=false",
        "enable_accel:=false",
        "align_depth.enable:=false",
        "pointcloud.enable:=false",
        "wait_for_device_timeout:=10.0",
    ]
    process = subprocess.Popen(
        command,
        cwd=REALWORLD,
        env=ros_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if D435I_RGB_TOPIC in ros_topic_names():
            if wait_for_rgb_frame(D435I_RGB_TOPIC, timeout=8.0):
                print(
                    f"D435i RGB-only ROS node started, pid={process.pid}, "
                    f"topic={D435I_RGB_TOPIC}, log={REALSENSE_LOG}"
                )
                return process
            break
        time.sleep(0.5)

    stop_realsense(process)
    tail = REALSENSE_LOG.read_text(errors="replace").splitlines()[-80:] if REALSENSE_LOG.exists() else []
    raise RuntimeError("D435i RGB node failed to start:\n" + "\n".join(tail))


def start_server(skip_clocks=False, model_checkpoint="MiniCPM-RobotTrack"):
    model_dir = resolve_model_checkpoint_path(model_checkpoint)
    print(f"Model checkpoint: {model_dir}")
    server_processes = matching_pids(SERVER_TOKEN)
    try:
        health = local_json(HEALTH_URL)
        if health.get("ok"):
            response_compatible = any(
                f"--response-mode {SERVER_RESPONSE_MODE}" in command
                for _pid, command in server_processes
            )
            running_model = str(health.get("model_dir", "") or "").strip()
            checkpoint_compatible = bool(running_model) and Path(running_model).resolve() == model_dir
            if response_compatible and checkpoint_compatible:
                print("Inference server is already healthy.")
                return
            if matching_pids(CLIENT_TOKEN):
                raise RuntimeError(
                    "Inference server must restart for the requested response mode/checkpoint. "
                    "Stop the active client first with: ./go2_runtime.py stop-control"
                )
            reasons = []
            if not response_compatible:
                reasons.append("response mode")
            if not checkpoint_compatible:
                reasons.append(f"checkpoint ({running_model or '<unknown>'} -> {model_dir})")
            print("Restarting inference server to change " + " and ".join(reasons) + ".")
            stop_server()
    except Exception:
        if matching_pids(CLIENT_TOKEN):
            raise

    if matching_pids(SERVER_TOKEN):
        stop_server()
    if not skip_clocks:
        print("Enabling jetson_clocks; enter the unitree sudo password if prompted.")
        subprocess.run(["sudo", "jetson_clocks"], check=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = SERVER_LOG.open("w")
    env = os.environ.copy()
    env.update(
        {
            "MINICPM_ROBOT_TRACK_ROOT": str(ROOT),
            "DINOV3_MODEL_PATH": str(ROOT / "minicpm_robot_track/backbones/dino_local_hf"),
            "SIGLIP_MODEL_PATH": str(ROOT / "minicpm_robot_track/backbones/siglip-so400m-patch14-384"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    command = [
        "python3", "sample/http_minicpm_robot_track_server.py",
        "--model_dir", str(model_dir),
        "--device", "cuda:0", "--host", "0.0.0.0", "--port", "5801", "--tcp-port", "5803",
        "--vision_amp", "bf16", "--planner_amp", "none",
        "--dino_backend", "trt_direct",
        "--dino_trt_engine", str(REALWORLD / "trt_artifacts/dino_patch_target_fp16.engine"),
        "--dino_trt_output_tokens", "576",
        "--siglip_backend", "trt_direct",
        "--siglip_trt_engine", str(REALWORLD / "trt_artifacts/siglip_pooled_target_maxn_fp16.engine"),
        "--siglip_trt_output_tokens", "576",
        "--response-mode", SERVER_RESPONSE_MODE, "--timing-mode", "fast", "--overlay-mode", "async",
    ]
    process = subprocess.Popen(
        command,
        cwd=REALWORLD,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    for _ in range(45):
        if process.poll() is not None:
            break
        try:
            if local_json(HEALTH_URL).get("ok"):
                print(f"Inference server started, pid={process.pid}, log={SERVER_LOG}")
                return
        except Exception:
            pass
        time.sleep(1.0)
    tail = SERVER_LOG.read_text(errors="replace").splitlines()[-80:] if SERVER_LOG.exists() else []
    raise RuntimeError("Inference server failed to start:\n" + "\n".join(tail))


def client_command(
    mode,
    instruction,
    camera_source="go2",
    waypoint_strategy="first",
    vx_positive_scale=1.0,
    vx_negative_scale=1.0,
    wz_scale=1.0,
    max_vx=0.15,
    max_wz=0.30,
    vx_positive_deadband=0.0,
    vx_negative_deadband=0.0,
    deadband_wz=0.0,
    hysteresis_vx=0.0,
    ema_alpha=1.0,
    yaw_boost=1.0,
    yaw_boost_threshold=0.10,
    yaw_boost_max=1.5,
    video_network_interface="enP8p1s0",
    video_client_timeout=3.0,
    jpeg_quality=60,
    jpeg_encoder="opencv",
    rgb_decode="fast",
    upload_width=384,
    model_input_mode="center_crop_height",
    model_crop_size=384,
    native_preview=True,
    camera_preview_fps=10.0,
    max_wait_new_frame_ms=35.0,
    fresh_frame_age_threshold_ms=0.0,
    latency_print_every=10,
    diagnostic_log_path="",
    record_videos=False,
    record_video_dir="",
    record_video_fps=15,
    record_video_segment_seconds=10,
):
    client_camera_source = "realsense" if camera_source == "d435i" else "go2"
    command = [
        "/usr/bin/python3", "-u", "client/http_minicpm_robot_track_client.py",
        "--server_url", "http://127.0.0.1:5801/eval_dual",
        "--transport", "tcp_jpeg", "--tcp_url", "tcp://127.0.0.1:5803",
        "--camera_source", client_camera_source,
        "--video_network_interface", str(video_network_interface),
        "--video_client_timeout", str(video_client_timeout), "--instruction", instruction,
        "--action_source", "server_velocity", "--jpeg_quality", str(jpeg_quality),
        "--jpeg_encoder", str(jpeg_encoder), "--rgb_decode", str(rgb_decode),
        "--upload_width", str(upload_width), "--model-input-mode", str(model_input_mode),
        "--model-crop-size", str(model_crop_size),
        "--camera-preview-fps", str(camera_preview_fps if native_preview else 0.0),
        "--max_wait_new_frame_ms", str(max_wait_new_frame_ms),
        "--fresh_frame_age_threshold_ms", str(fresh_frame_age_threshold_ms),
        "--latency_print_every", str(latency_print_every), "--log_path", str(REALWORLD / "output"),
        "--waypoint-strategy", waypoint_strategy,
        "--vx-positive-scale", str(vx_positive_scale),
        "--vx-negative-scale", str(vx_negative_scale),
        "--wz-scale", str(wz_scale),
        "--vx-positive-deadband", str(vx_positive_deadband),
        "--vx-negative-deadband", str(vx_negative_deadband),
        "--cmd-deadband-w", str(deadband_wz),
        "--cmd-max-v", str(max_vx), "--cmd-max-w", str(max_wz),
        "--cmd-hysteresis-v", str(hysteresis_vx), "--cmd-ema-alpha", str(ema_alpha),
        "--yaw-boost", str(yaw_boost),
        "--yaw-boost-threshold", str(yaw_boost_threshold),
        "--yaw-boost-max", str(yaw_boost_max),
    ]
    if diagnostic_log_path:
        command += ["--diagnostic-log-path", str(diagnostic_log_path)]
    if record_videos:
        command += [
            "--record-video-pair",
            "--record-video-dir", str(record_video_dir or (REALWORLD / "output/videos")),
            "--record-video-fps", str(record_video_fps),
            "--record-video-segment-seconds", str(record_video_segment_seconds),
        ]
    log_prefix = "dryrun" if mode == "dry-run" else "live"
    camera_log_part = "" if camera_source == "go2" else f"-{camera_source}"
    strategy_log = REALWORLD / f"output/{log_prefix}{camera_log_part}-{waypoint_strategy}-latest.json"
    latest_alias = REALWORLD / f"output/{log_prefix}-latest.json"
    if mode == "dry-run":
        command += [
            "--latency_log_path", str(strategy_log),
            "--latency_log_alias", str(latest_alias),
            "--dry_run",
        ]
    else:
        command += [
            "--latency_log_path", str(strategy_log),
            "--latency_log_alias", str(latest_alias),
            "--enable-live-control",
        ]
    return command


def run(
    mode,
    instruction,
    camera_source,
    waypoint_strategy,
    detailed_logging,
    model_checkpoint,
    vx_positive_scale,
    vx_negative_scale,
    wz_scale,
    confirm_live,
    skip_jetson_clocks,
    max_vx,
    max_wz,
    vx_positive_deadband,
    vx_negative_deadband,
    deadband_wz,
    hysteresis_vx,
    ema_alpha,
    yaw_boost,
    yaw_boost_threshold,
    yaw_boost_max,
    video_network_interface,
    video_client_timeout,
    jpeg_quality,
    jpeg_encoder,
    rgb_decode,
    upload_width,
    model_input_mode,
    model_crop_size,
    native_preview,
    camera_preview_fps,
    max_wait_new_frame_ms,
    fresh_frame_age_threshold_ms,
    latency_print_every,
    record_videos,
    record_video_fps,
    record_video_segment_seconds,
):
    if matching_pids(CLIENT_TOKEN):
        raise RuntimeError("A camera/control client is already running. Use: go2_runtime.py stop-control")
    if mode == "live":
        if not confirm_live:
            raise RuntimeError("Live mode requires --confirm-live-control")
        if not sys.stdin.isatty():
            raise RuntimeError("Live mode must be started in an interactive terminal")
        print("LIVE CONTROL: clear the area and keep a second terminal ready with the stop-control command.")
        if input("Type MOVE to continue: ").strip() != "MOVE":
            print("Cancelled.")
            return
        print(
            f"Live command limits: vx=+/-{max_vx:.2f} m/s, vy=0 (lateral disabled), "
            f"wz=+/-{max_wz:.2f} rad/s. "
            f"vx_scale=(+{vx_positive_scale:.2f},-{vx_negative_scale:.2f}), "
            f"wz_scale={wz_scale:.2f}, "
            f"vx_deadband=(+{vx_positive_deadband:.2f},-{vx_negative_deadband:.2f}), "
            f"wz_deadband={deadband_wz:.2f}, "
            f"hysteresis_vx={hysteresis_vx:.2f}, ema_alpha={ema_alpha:.2f}."
        )
    model_dir = resolve_model_checkpoint_path(model_checkpoint)
    print(f"Camera source: {camera_source}")
    print(f"Model checkpoint: {model_dir}")
    print(
        "Native camera preview: "
        + (f"enabled at {camera_preview_fps:.1f} FPS" if native_preview and camera_preview_fps > 0.0 else "disabled")
    )
    print(f"Waypoint strategy: {waypoint_strategy}")
    print(
        f"Velocity scale: vx_positive={vx_positive_scale:.3f}, "
        f"vx_negative={vx_negative_scale:.3f}, wz={wz_scale:.3f}"
    )
    print(
        f"Velocity deadband: vx_positive={vx_positive_deadband:.3f}, "
        f"vx_negative={vx_negative_deadband:.3f}, wz={deadband_wz:.3f}"
    )
    client_log = None
    diagnostic_log = ""
    tegrastats_process = None
    session_dir = None
    if detailed_logging:
        run_config = {
            "mode": mode,
            "instruction": instruction,
            "camera_source": camera_source,
            "waypoint_strategy": waypoint_strategy,
            "model_checkpoint": str(model_dir),
            "skip_jetson_clocks": skip_jetson_clocks,
            "vx_positive_scale": vx_positive_scale,
            "vx_negative_scale": vx_negative_scale,
            "wz_scale": wz_scale,
            "max_vx": max_vx,
            "max_wz": max_wz,
            "vx_positive_deadband": vx_positive_deadband,
            "vx_negative_deadband": vx_negative_deadband,
            "deadband_wz": deadband_wz,
            "hysteresis_vx": hysteresis_vx,
            "ema_alpha": ema_alpha,
            "yaw_boost": yaw_boost,
            "yaw_boost_threshold": yaw_boost_threshold,
            "yaw_boost_max": yaw_boost_max,
            "video_network_interface": video_network_interface,
            "video_client_timeout": video_client_timeout,
            "jpeg_quality": jpeg_quality,
            "jpeg_encoder": jpeg_encoder,
            "rgb_decode": rgb_decode,
            "upload_width": upload_width,
            "model_input_mode": model_input_mode,
            "model_crop_size": model_crop_size,
            "native_preview": native_preview,
            "camera_preview_fps": camera_preview_fps,
            "effective_camera_preview_fps": camera_preview_fps if native_preview else 0.0,
            "max_wait_new_frame_ms": max_wait_new_frame_ms,
            "fresh_frame_age_threshold_ms": fresh_frame_age_threshold_ms,
            "latency_print_every": latency_print_every,
            "record_videos": record_videos,
            "record_video_fps": record_video_fps,
            "record_video_segment_seconds": record_video_segment_seconds,
        }
        session_dir = create_run_log_bundle(mode, camera_source, waypoint_strategy, run_config)
        client_log = session_dir / "client.log"
        diagnostic_log = session_dir / "diagnostics.jsonl"
        tegrastats_log = session_dir / "tegrastats.log"
        tegrastats_process = start_tegrastats_log(tegrastats_log)
        print(f"Persistent run logs: {session_dir}")
    record_video_dir = (
        session_dir / "videos"
        if session_dir is not None
        else REALWORLD / "output/videos"
    )
    realsense_process = None
    try:
        if camera_source == "go2":
            go2_camera_preflight(video_network_interface)
        else:
            realsense_process = start_realsense_rgb()
        start_server(skip_clocks=skip_jetson_clocks, model_checkpoint=str(model_dir))
        command = client_command(
            mode,
            instruction,
            camera_source=camera_source,
            waypoint_strategy=waypoint_strategy,
            vx_positive_scale=vx_positive_scale,
            vx_negative_scale=vx_negative_scale,
            wz_scale=wz_scale,
            max_vx=max_vx,
            max_wz=max_wz,
            vx_positive_deadband=vx_positive_deadband,
            vx_negative_deadband=vx_negative_deadband,
            deadband_wz=deadband_wz,
            hysteresis_vx=hysteresis_vx,
            ema_alpha=ema_alpha,
            yaw_boost=yaw_boost,
            yaw_boost_threshold=yaw_boost_threshold,
            yaw_boost_max=yaw_boost_max,
            video_network_interface=video_network_interface,
            video_client_timeout=video_client_timeout,
            jpeg_quality=jpeg_quality,
            jpeg_encoder=jpeg_encoder,
            rgb_decode=rgb_decode,
            upload_width=upload_width,
            model_input_mode=model_input_mode,
            model_crop_size=model_crop_size,
            native_preview=native_preview,
            camera_preview_fps=camera_preview_fps,
            max_wait_new_frame_ms=max_wait_new_frame_ms,
            fresh_frame_age_threshold_ms=fresh_frame_age_threshold_ms,
            latency_print_every=latency_print_every,
            diagnostic_log_path=diagnostic_log,
            record_videos=record_videos,
            record_video_dir=record_video_dir,
            record_video_fps=record_video_fps,
            record_video_segment_seconds=record_video_segment_seconds,
        )
        max_attempts = 3 if mode == "dry-run" else 1
        for attempt in range(1, max_attempts + 1):
            print(
                f"Starting {mode} camera client"
                f" (attempt {attempt}/{max_attempts}). Press Ctrl-C to stop control."
            )
            started = time.monotonic()
            if detailed_logging:
                returncode = run_client_with_tee(
                    command,
                    cwd=REALWORLD,
                    env=ros_env(),
                    log_path=client_log,
                )
            else:
                result = subprocess.run(command, cwd=REALWORLD, env=ros_env(), check=False)
                returncode = result.returncode
            elapsed = time.monotonic() - started
            if returncode == 0:
                break
            can_retry = attempt < max_attempts and elapsed < 15.0
            if not can_retry:
                raise RuntimeError(
                    f"Camera client exited with code {returncode} after {elapsed:.1f}s"
                )
            print(
                f"Camera client failed during startup (code={returncode}, "
                f"elapsed={elapsed:.1f}s). Rechecking DDS/camera before retry."
            )
            time.sleep(1.0)
            if camera_source == "go2":
                go2_camera_preflight(video_network_interface)
            elif not wait_for_rgb_frame(D435I_RGB_TOPIC):
                raise RuntimeError(f"D435i stopped publishing {D435I_RGB_TOPIC}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_control(send_stop=(mode == "live"), network_interface=video_network_interface)
        stop_realsense(realsense_process)
        stop_background_process(tegrastats_process)


def status():
    print("Inference server:")
    for pid, command in matching_pids(SERVER_TOKEN):
        print(f"  pid={pid} {command}")
    if not matching_pids(SERVER_TOKEN):
        print("  not running")
    print("Camera/control client:")
    for pid, command in matching_pids(CLIENT_TOKEN):
        print(f"  pid={pid} {command}")
    if not matching_pids(CLIENT_TOKEN):
        print("  not running")
    print("RealSense camera:")
    for pid, command in matching_pids(REALSENSE_TOKEN):
        print(f"  pid={pid} {command}")
    if not matching_pids(REALSENSE_TOKEN):
        print("  not running")
    if USB_C_ROLE.exists():
        print(f"  Orin USB-C role: {USB_C_ROLE.read_text().strip()}")
    for label, url in (("health", HEALTH_URL), ("state", STATE_URL)):
        try:
            print(f"{label}: {json.dumps(local_json(url), ensure_ascii=False)}")
        except Exception as exc:
            print(f"{label}: unavailable ({exc})")


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-RobotTrack Go2 runtime manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start the server if needed, then run the camera client")
    run_parser.add_argument("--config", default=str(DEFAULT_RUN_CONFIG_PATH), help="Runtime YAML configuration")
    run_parser.add_argument("--mode", choices=["dry-run", "live"], default=None)
    run_parser.add_argument("--instruction", default=None)
    run_parser.add_argument(
        "--camera-source",
        choices=["go2", "d435i"],
        default=None,
        help="Override the YAML RGB source.",
    )
    run_parser.add_argument(
        "--waypoint-strategy",
        choices=WAYPOINT_STRATEGIES,
        default=None,
        help="Override the YAML waypoint selection/latency strategy.",
    )
    run_parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Checkpoint name under minicpm_robot_track/checkpoints, or an absolute HF checkpoint path.",
    )
    run_parser.add_argument("--confirm-live-control", action="store_true")
    run_parser.add_argument(
        "--detailed-logging",
        action="store_true",
        help="Persist 10 Hz control diagnostics, client output, and tegrastats for one test run.",
    )
    run_parser.add_argument(
        "--skip-jetson-clocks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override runtime.skip_jetson_clocks from YAML.",
    )
    run_parser.add_argument("--vx-scale", type=float, default=None, help="Set both positive and negative vx scales")
    run_parser.add_argument("--vx-positive-scale", type=float, default=None)
    run_parser.add_argument("--vx-negative-scale", type=float, default=None)
    run_parser.add_argument("--wz-scale", type=float, default=None)
    run_parser.add_argument("--max-vx", type=float, default=None)
    run_parser.add_argument("--max-wz", type=float, default=None)
    run_parser.add_argument("--deadband-vx", type=float, default=None, help="Set both positive and negative vx deadbands")
    run_parser.add_argument("--vx-positive-deadband", type=float, default=None)
    run_parser.add_argument("--vx-negative-deadband", type=float, default=None)
    run_parser.add_argument("--deadband-wz", type=float, default=None)
    run_parser.add_argument("--hysteresis-vx", type=float, default=None)
    run_parser.add_argument("--ema-alpha", type=float, default=None)
    run_parser.add_argument("--yaw-boost", type=float, default=None)
    run_parser.add_argument("--yaw-boost-threshold", type=float, default=None)
    run_parser.add_argument("--yaw-boost-max", type=float, default=None)
    run_parser.add_argument("--video-network-interface", default=None)
    run_parser.add_argument("--video-client-timeout", type=float, default=None)
    run_parser.add_argument("--jpeg-quality", type=int, default=None)
    run_parser.add_argument("--jpeg-encoder", choices=["opencv", "pil"], default=None)
    run_parser.add_argument("--rgb-decode", choices=["fast", "cv_bridge"], default=None)
    run_parser.add_argument("--upload-width", type=int, default=None)
    run_parser.add_argument(
        "--model-input-mode",
        choices=["aspect_resize", "center_crop_height", "center_crop_720p"],
        default=None,
    )
    run_parser.add_argument("--model-crop-size", type=int, default=None)
    run_parser.add_argument(
        "--native-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish the native camera stream to /stream/camera; use --no-native-preview to disable it.",
    )
    run_parser.add_argument("--camera-preview-fps", type=float, default=None)
    run_parser.add_argument("--max-wait-new-frame-ms", type=float, default=None)
    run_parser.add_argument("--fresh-frame-age-threshold-ms", type=float, default=None)
    run_parser.add_argument("--latency-print-every", type=int, default=None)
    run_parser.add_argument(
        "--record-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record native RGB plus the 384x384 waypoint/velocity overlay.",
    )
    run_parser.add_argument("--record-video-fps", type=int, default=None)
    run_parser.add_argument("--record-video-segment-seconds", type=int, default=None)

    server_parser = subparsers.add_parser("server", help="Start only the inference server")
    server_parser.add_argument("--skip-jetson-clocks", action="store_true")
    server_parser.add_argument("--model-checkpoint", default=RUN_CONFIG_DEFAULTS["model_checkpoint"])
    subparsers.add_parser("status", help="Show server, client and web state")
    for name, help_text in (
        ("stop-control", "Stop the client and send Unitree StopMove"),
        ("stop", "Stop control and the inference server"),
    ):
        stop_parser = subparsers.add_parser(name, help=help_text)
        stop_parser.add_argument("--config", default=str(DEFAULT_RUN_CONFIG_PATH))
        stop_parser.add_argument("--video-network-interface", default=None)

    args = parser.parse_args()
    if args.command == "run":
        config = resolve_run_config(args)
        print(f"Runtime config: {Path(args.config).expanduser()}")
        run(
            confirm_live=args.confirm_live_control,
            detailed_logging=args.detailed_logging,
            **config,
        )
    elif args.command == "server":
        start_server(skip_clocks=args.skip_jetson_clocks, model_checkpoint=args.model_checkpoint)
    elif args.command == "status":
        status()
    elif args.command == "stop-control":
        interface = args.video_network_interface or load_run_config(args.config)["video_network_interface"]
        stop_control(send_stop=True, network_interface=interface)
    elif args.command == "stop":
        interface = args.video_network_interface or load_run_config(args.config)["video_network_interface"]
        stop_control(send_stop=True, network_interface=interface)
        stop_server()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
