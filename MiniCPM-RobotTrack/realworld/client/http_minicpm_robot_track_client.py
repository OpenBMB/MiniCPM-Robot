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
import atexit
import copy
import ctypes
import io
import json
import math
import multiprocessing as mp
import os
import socket
import struct
import sys
import threading
import textwrap
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
import requests
from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import CameraInfo, Image

frame_data = {}
frame_idx = 0
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock
from waypoint_selection import (
    WAYPOINT_STRATEGIES,
    apply_directional_vx_deadband,
    scale_velocity_command,
    select_waypoint_velocity,
    two_step_velocity_at_elapsed,
    two_step_velocity_plan,
)


# global variable
policy_init = True
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None

desired_v, desired_w = 0.0, 0.0
exit_requested = False
video_recorder = None
waypoint_recorder = None
native_video_recorder = None
control_video_recorder = None
latest_waypoints = []
latest_base_velocity = [0.0, 0.0, 0.0]
latest_demo_frame = None
latest_demo_step = 0
latest_demo_infer = 0.0
demo_overlay_lock = threading.Lock()
latency_samples = []
latency_samples_lock = threading.Lock()
diagnostic_queue = deque()
diagnostic_queue_cv = threading.Condition()
diagnostic_writer_thread = None
diagnostic_writer_stop = False
DIAGNOSTIC_QUEUE_LIMIT = 8192
LOG_PATH = None
rgb_depth_rw_lock = ReadWriteLock()
rgb_frame_cv = threading.Condition()
odom_rw_lock = ReadWriteLock()
video_overlay_lock = threading.Lock()
latest_log_output = {"status": "waiting_first_response"}
latest_log_step = 0
latest_log_infer_time = 0.0
latest_latency_metrics = {"status": "waiting_first_response"}
http_session = None
tcp_eval_client = None
TCP_REQ_HEADER = struct.Struct("!4sII")
TCP_RESP_HEADER = struct.Struct("!4sI")
TCP_REQ_MAGIC = b"OVL1"
TCP_RESP_MAGIC = b"OVR1"

# ── Step 2: 50 Hz cmd dispatch ────────────────────────────────────────
# Single writer to manager.move(). planning_thread / control_thread only
# *publish* a plan into current_plan; the dispatch thread rolls it into a
# fresh (v, w) every 20 ms. This decouples planner cadence (~1.4 Hz) from
# control loop cadence (50 Hz).
DISPATCH_HZ = 50
MAX_PLAN_AGE_S = 1.5  # safety: drop to (0, 0) if no fresh plan within this window
plan_lock = threading.Lock()
current_plan = {
    "kind": None,            # None | "server_velocity" | "two_step"
    "opt_u": None,           # single (v, w) or a two-step plan
    "arrival_time": 0.0,
    "timeline_start": 0.0,   # rollout: image timestamp; otherwise arrival_time
    "dt": 0.1,
    "horizon": 1,
    "version": 0,
}

# ── Step 1: instrumentation globals ───────────────────────────────────
last_dispatch_time = 0.0
last_inter_update_ms = 0.0
last_commanded_v = 0.0
last_commanded_w = 0.0
# raw = before deadband/EMA; final = after. Useful for analyze_log diagnostics.
last_raw_v = 0.0
last_raw_w = 0.0
# EMA / hysteresis state (touched only by cmd_dispatch_thread)
_ema_v = 0.0
_ema_w = 0.0
_prev_sign_v = 0  # -1, 0, +1


def queue_diagnostic_event(event, **fields):
    """Queue one JSONL diagnostic event without blocking the control loop on I/O."""
    if not getattr(client_args, "diagnostic_log_path", ""):
        return
    payload = {"event": str(event), "t": time.time(), **fields}
    with diagnostic_queue_cv:
        if len(diagnostic_queue) >= DIAGNOSTIC_QUEUE_LIMIT:
            diagnostic_queue.popleft()
        diagnostic_queue.append(payload)
        diagnostic_queue_cv.notify()


def _diagnostic_writer_loop(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_sync = time.monotonic()
    try:
        with open(path, "a", buffering=1) as log_file:
            while True:
                with diagnostic_queue_cv:
                    if not diagnostic_queue and not diagnostic_writer_stop:
                        diagnostic_queue_cv.wait(timeout=0.25)
                    pending = list(diagnostic_queue)
                    diagnostic_queue.clear()
                    should_stop = diagnostic_writer_stop and not pending
                for payload in pending:
                    log_file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                now = time.monotonic()
                if pending:
                    log_file.flush()
                if pending and now - last_sync >= 1.0:
                    os.fsync(log_file.fileno())
                    last_sync = now
                if should_stop:
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    return
    except Exception as exc:
        print(f"[diagnostic] writer failed for {path}: {exc}", flush=True)


def start_diagnostic_writer(path):
    global diagnostic_writer_thread, diagnostic_writer_stop
    if not path or diagnostic_writer_thread is not None:
        return
    diagnostic_writer_stop = False
    diagnostic_writer_thread = threading.Thread(
        target=_diagnostic_writer_loop,
        args=(str(path),),
        name="diagnostic-jsonl-writer",
        daemon=True,
    )
    diagnostic_writer_thread.start()
    atexit.register(stop_diagnostic_writer)
    queue_diagnostic_event(
        "session_start",
        pid=os.getpid(),
        argv=sys.argv,
        camera_source=getattr(client_args, "camera_source", ""),
        waypoint_strategy=_effective_waypoint_strategy(),
        video_network_interface=getattr(client_args, "video_network_interface", ""),
    )
    print(f"[diagnostic] incremental JSONL -> {path}", flush=True)


def stop_diagnostic_writer():
    global diagnostic_writer_thread, diagnostic_writer_stop
    writer = diagnostic_writer_thread
    if writer is None:
        return
    queue_diagnostic_event("session_stop", exit_requested=bool(exit_requested))
    with diagnostic_queue_cv:
        diagnostic_writer_stop = True
        diagnostic_queue_cv.notify_all()
    if writer is not threading.current_thread():
        writer.join(timeout=3.0)
    diagnostic_writer_thread = None


def prepare_center_crop_height(raw_rgb, crop_size=384):
    """Resize an RGB frame to crop_size high, then center-crop crop_size square."""
    if raw_rgb is None or raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB frame")
    src_h, src_w = raw_rgb.shape[:2]
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop size must be positive")
    target_h = crop_size
    target_w = max(1, int(round(src_w * (target_h / float(src_h)))))
    if target_w < crop_size:
        target_w = crop_size
        target_h = max(1, int(round(src_h * (target_w / float(src_w)))))
    resized = cv2.resize(raw_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if target_w < crop_size or target_h < crop_size:
        raise ValueError(
            f"resized frame {target_w}x{target_h} is smaller than crop {crop_size}x{crop_size}"
        )
    x0 = (target_w - crop_size) // 2
    y0 = (target_h - crop_size) // 2
    crop = np.ascontiguousarray(resized[y0:y0 + crop_size, x0:x0 + crop_size])
    return crop, (target_w, target_h, x0, y0)


def encode_camera_preview_jpeg(raw_rgb, source_jpeg=None, quality=60):
    """Return a native source JPEG, or encode an RGB frame for web preview."""
    if source_jpeg:
        return bytes(source_jpeg)
    if raw_rgb is None or raw_rgb.ndim != 3 or raw_rgb.shape[2] != 3:
        return None
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))],
    )
    return encoded.tobytes() if ok else None


def _publish_plan(kind, opt_u, dt=0.1, horizon=1, timeline_start=None):
    """Thread-safe plan publish. Always pair with arrival_time bumped to now."""
    arrival_time = time.time()
    with plan_lock:
        current_plan["kind"] = kind
        current_plan["opt_u"] = opt_u
        current_plan["arrival_time"] = arrival_time
        current_plan["timeline_start"] = (
            float(timeline_start) if timeline_start is not None else arrival_time
        )
        current_plan["dt"] = float(dt)
        current_plan["horizon"] = int(horizon)
        current_plan["version"] += 1


def _effective_waypoint_strategy():
    return str(getattr(client_args, "waypoint_strategy", "first")).strip().lower()


def _response_waypoints(response):
    waypoints = response.get("waypoints", [])
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        return None
    if not all(isinstance(waypoint, list) and len(waypoint) >= 3 for waypoint in waypoints):
        return None
    return waypoints


def _select_server_waypoint_action(response, base_velocity, rgb_time, frame_pose):
    """Select one delayed waypoint command or build a timestamped rollout."""
    raw_v = float(base_velocity[0]) if len(base_velocity) > 0 else 0.0
    raw_w = float(base_velocity[2]) if len(base_velocity) > 2 else 0.0
    strategy = _effective_waypoint_strategy()
    now = time.time()
    age_s = max(0.0, now - float(rgb_time))
    control_dt = max(1e-6, float(response.get("control_dt", 0.1) or 0.1))
    waypoints = _response_waypoints(response)
    info = {
        "waypoint_strategy": strategy,
        "waypoint_count": len(waypoints) if waypoints is not None else 0,
        "waypoint_age_ms": age_s * 1000.0,
        "waypoint_latency_steps": age_s / control_dt,
        "waypoint_selected_index": 1 if strategy == "first" else -1,
        "waypoint_selected_segment": -1,
        "waypoint_blend_alpha": 0.0,
        "waypoint_plan_expired": 0.0,
        "waypoint_selection_source": "server_velocity",
    }

    if strategy == "first":
        return raw_v, raw_w, info, None

    if waypoints is None:
        info["waypoint_selection_source"] = "server_velocity_missing_waypoints"
        return raw_v, raw_w, info, None

    if strategy == "two-step":
        rollout = np.asarray(two_step_velocity_plan(waypoints, control_dt), dtype=np.float64)
        info.update({
            "waypoint_selected_segment": 0,
            "waypoint_selected_index": 1,
            "waypoint_selection_source": "two_step_arrival_timeline",
        })
        return float(rollout[0, 0]), float(rollout[0, 1]), info, {
            "kind": "two_step",
            "opt_u": np.ascontiguousarray(rollout),
            "dt": control_dt,
            "horizon": int(rollout.shape[0]),
        }

    v, w, selection_info = select_waypoint_velocity(
        waypoints=waypoints,
        strategy=strategy,
        age_s=age_s,
        control_dt=control_dt,
    )
    info.update(selection_info)
    info["waypoint_selection_source"] = "client_waypoints"
    return float(v), float(w), info, None


class TcpJpegEvalClient:
    def __init__(self, tcp_url):
        from urllib.parse import urlparse

        if "://" not in tcp_url:
            tcp_url = "tcp://" + tcp_url
        u = urlparse(tcp_url)
        if not u.hostname or not u.port:
            raise ValueError(f"tcp_url must look like tcp://host:port, got {tcp_url!r}")
        self.host = u.hostname
        self.port = int(u.port)
        self.sock = None
        self._connect()

    def _connect(self):
        self.close()
        s = socket.create_connection((self.host, self.port), timeout=10)
        s.settimeout(30)
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.sock = s
        print(f"[tcp] connected to tcp://{self.host}:{self.port}")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _recv_exact(self, nbytes):
        buf = bytearray(nbytes)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            n = self.sock.recv_into(view[got:], nbytes - got)
            if n == 0:
                raise ConnectionError("TCP inference server closed the connection")
            got += n
        return bytes(buf)

    def request(self, payload, image_bytes):
        json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = TCP_REQ_HEADER.pack(TCP_REQ_MAGIC, len(json_bytes), len(image_bytes))
        for attempt in range(2):
            try:
                if self.sock is None:
                    self._connect()
                self.sock.sendall(header + json_bytes + image_bytes)
                magic, resp_len = TCP_RESP_HEADER.unpack(self._recv_exact(TCP_RESP_HEADER.size))
                if magic != TCP_RESP_MAGIC:
                    raise RuntimeError(f"bad TCP response magic: {magic!r}")
                if resp_len <= 0 or resp_len > (8 << 20):
                    raise RuntimeError(f"bad TCP response length: {resp_len}")
                out = json.loads(self._recv_exact(resp_len).decode("utf-8"))
                if "error" in out:
                    raise RuntimeError(str(out["error"]))
                return out
            except Exception:
                self.close()
                if attempt == 0:
                    self._connect()
                    continue
                raise


class VideoRecorder:
    def __init__(self, out_path, fps=24, segment_seconds=0.0):
        self.out_path = out_path
        self.fps = fps
        self.segment_seconds = max(0.0, float(segment_seconds))
        self.segment_index = 0
        self.segment_started = 0.0
        self.writer = None
        self.writer_size = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame_rgb):
        if frame_rgb is None:
            return
        frame = np.ascontiguousarray(frame_rgb.astype(np.uint8))
        with self.lock:
            self.latest_frame = frame

    def _segment_path(self):
        if self.segment_seconds <= 0.0:
            return self.out_path
        stem, suffix = os.path.splitext(self.out_path)
        return f"{stem}-part{self.segment_index:03d}{suffix or '.mp4'}"

    def _init_writer(self, frame):
        h, w = frame.shape[:2]
        self.writer_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = self._segment_path()
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.writer_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
        self.segment_started = time.monotonic()

    def _rotate_segment_if_needed(self, frame):
        if (
            self.writer is None
            or self.segment_seconds <= 0.0
            or time.monotonic() - self.segment_started < self.segment_seconds
        ):
            return
        self.writer.release()
        self.writer = None
        self.segment_index += 1
        self._init_writer(frame)

    def _loop(self):
        frame_period = 1.0 / float(self.fps)
        next_ts = time.time()
        while self.running:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                if self.writer is None:
                    self._init_writer(frame)
                else:
                    self._rotate_segment_if_needed(frame)
                if (frame.shape[1], frame.shape[0]) != self.writer_size:
                    frame = cv2.resize(frame, self.writer_size, interpolation=cv2.INTER_AREA)
                self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            next_ts += frame_period
            sleep_time = next_ts - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_ts = time.time()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def compose_log_frame(rgb_np, depth_np, instruction, json_output, step_idx, infer_time, latency_metrics=None):
    """Compose a frame similar to server log image: RGB | Depth + text panel."""
    if rgb_np is None or depth_np is None:
        return None

    rgb_np = np.asarray(rgb_np)
    depth_np = np.asarray(depth_np)
    if rgb_np.ndim != 3:
        return None

    h, w = rgb_np.shape[:2]
    rgb_pil = PIL_Image.fromarray(rgb_np.astype(np.uint8))

    depth_vis = depth_np.astype(np.float32).copy()
    dmax = float(depth_vis.max()) if depth_vis.size > 0 else 0.0
    if dmax > 1e-6:
        depth_vis = (depth_vis / dmax * 255.0).astype(np.uint8)
    else:
        depth_vis = np.zeros((h, w), dtype=np.uint8)
    depth_pil = PIL_Image.fromarray(depth_vis).convert("RGB")

    text_h = 140
    canvas = PIL_Image.new("RGB", (w * 2, h + text_h), (30, 30, 30))
    canvas.paste(rgb_pil, (0, 0))
    canvas.paste(depth_pil, (w, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    wrapped_instruction = textwrap.fill(instruction, width=100)
    output_str = json.dumps(json_output, ensure_ascii=False)
    if len(output_str) > 200:
        output_str = output_str[:200] + "..."
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    timing = json_output.get("timing", {}) if isinstance(json_output, dict) else {}
    latency_text = ""
    if isinstance(latency_metrics, dict) and latency_metrics:
        latency_text = (
            f"\ncam->action={latency_metrics.get('camera_to_action_ms', 0.0):.1f}ms "
            f"frame_age={latency_metrics.get('frame_age_ms', 0.0):.1f}ms "
            f"(cb={latency_metrics.get('callback_lag_ms', 0.0):.1f} "
            f"queue={latency_metrics.get('queue_wait_ms', 0.0):.1f}) "
            f"http={latency_metrics.get('client_http_roundtrip_ms', 0.0):.1f}ms "
            f"post_http={latency_metrics.get('post_http_to_action_ms', 0.0):.1f}ms"
        )

    text = (
        f"Step: {step_idx}  |  Infer: {infer_time:.2f}s  |  Time: {timestamp}\n"
        f"Instruction: {wrapped_instruction}\n"
        f"Output: {output_str}{latency_text}\n"
        f"Timing: vision={float(timing.get('vision_encode_ms', 0.0)):.1f}ms "
        f"policy={float(timing.get('policy_forward_ms', 0.0)):.1f}ms "
        f"llm={float(timing.get('llm_backbone_ms', 0.0)):.1f}ms"
    )
    draw.text((10, h + 5), text, fill=(220, 220, 220), font=font)
    draw.text((5, 5), "RGB", fill=(0, 255, 0), font=font)
    draw.text((w + 5, 5), "Depth", fill=(0, 255, 0), font=font)

    return np.asarray(canvas)


def compose_waiting_frame(instruction, width=640, height=480):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    return compose_log_frame(rgb, depth, instruction, {"status": "waiting_first_response"}, 0, 0.0)


def update_video_overlay(json_output, step_idx, infer_time, latency_metrics=None):
    global latest_log_output, latest_log_step, latest_log_infer_time, latest_latency_metrics
    with video_overlay_lock:
        latest_log_output = copy.deepcopy(json_output)
        latest_log_step = int(step_idx)
        latest_log_infer_time = float(infer_time)
        if latency_metrics is not None:
            latest_latency_metrics = copy.deepcopy(latency_metrics)


def video_render_thread():
    """Render RGB/Depth at fixed FPS and keep text panel synced to latest server result."""
    global manager
    frame_period = 1.0 / 24.0
    next_ts = time.time()
    instruction = client_args.instruction

    while not exit_requested:
        if video_recorder is None or not video_recorder.running:
            time.sleep(0.05)
            continue

        rgb_for_log = None
        depth_for_log = None
        if manager is not None:
            rgb_depth_rw_lock.acquire_read()
            if manager.rgb_image is not None:
                rgb_for_log = copy.deepcopy(manager.rgb_image)
            if manager.depth_image is not None:
                depth_for_log = copy.deepcopy(manager.depth_image)
            rgb_depth_rw_lock.release_read()

        with video_overlay_lock:
            json_output = copy.deepcopy(latest_log_output)
            step_idx = latest_log_step
            infer_time = latest_log_infer_time
            latency_metrics = copy.deepcopy(latest_latency_metrics)

        if rgb_for_log is None or depth_for_log is None:
            frame = compose_waiting_frame(instruction)
        else:
            frame = compose_log_frame(
                rgb_for_log,
                depth_for_log,
                instruction,
                json_output,
                step_idx,
                infer_time,
                latency_metrics=latency_metrics,
            )

        if frame is not None:
            video_recorder.update_frame(frame)

        next_ts += frame_period
        sleep_time = next_ts - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_ts = time.time()


def draw_waypoint_overlay(frame_rgb, waypoints, base_velocity, step, infer_time):
    """Overlay predicted waypoints + velocity panel onto a copy of frame_rgb."""
    if frame_rgb is None:
        return None
    base = cv2.cvtColor(np.ascontiguousarray(frame_rgb.astype(np.uint8)), cv2.COLOR_RGB2BGR)
    h, w = base.shape[:2]
    panel_h = 130
    vis = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    vis[:h, :, :] = base
    cv2.rectangle(vis, (0, h), (w - 1, h + panel_h - 1), (20, 20, 20), thickness=-1)
    cv2.line(vis, (0, h), (w - 1, h), (90, 90, 90), 1)

    vx = float(base_velocity[0]) if len(base_velocity) > 0 else 0.0
    vy = float(base_velocity[1]) if len(base_velocity) > 1 else 0.0
    wz = float(base_velocity[2]) if len(base_velocity) > 2 else 0.0

    panel_w = min(520, max(360, w - 20))
    panel_x0 = (w - panel_w) // 2
    panel_x1 = panel_x0 + panel_w
    panel_y0, panel_y1 = h + 12, h + panel_h - 10
    cv2.rectangle(vis, (panel_x0, panel_y0), (panel_x1, panel_y1), (25, 25, 25), thickness=-1)
    cv2.rectangle(vis, (panel_x0, panel_y0), (panel_x1, panel_y1), (220, 220, 220), thickness=1)

    cx, cy = (panel_x0 + panel_x1) // 2, panel_y0 + 58
    sx = 60.0 / 0.8
    sy = 60.0 / 1.2
    dx = int(np.clip(-vy * sx, -70, 70))
    dy = int(np.clip(-vx * sy, -70, 70))
    cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy), (30, 80, 255), 3, tipLength=0.25)
    cv2.circle(vis, (cx, cy), 2, (230, 230, 230), -1)

    traj_xy = []
    for wp in waypoints or []:
        if not isinstance(wp, (list, tuple)) or len(wp) < 2:
            continue
        try:
            traj_xy.append((float(wp[0]), float(wp[1])))
        except Exception:
            continue
    if traj_xy:
        max_abs_x = max(abs(p[0]) for p in traj_xy)
        max_abs_y = max(abs(p[1]) for p in traj_xy)
        traj_scale = 58.0 / max(0.20, max_abs_x, max_abs_y)
        pts = []
        for wx, wy in traj_xy:
            px = int(np.clip(cx - wy * traj_scale, panel_x0 + 8, panel_x1 - 8))
            py = int(np.clip(cy - wx * traj_scale, panel_y0 + 10, panel_y1 - 10))
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 200, 255), thickness=2)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 3 if i == 0 else 2, (0, 255, 255), -1)

    turn = "left" if wz > 0.05 else ("right" if wz < -0.05 else "straight")
    cv2.putText(vis, f"step={step} infer={infer_time:.2f}s",
                (panel_x0 + 10, panel_y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}",
                (panel_x0 + 10, panel_y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, f"turn={turn} (|wz|max=1.50)",
                (panel_x0 + 10, panel_y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis, "Linear dir: up=+vx, left=+vy | Yellow: predicted waypoints",
                (panel_x0 + 10, panel_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def update_demo_overlay(response, step_idx, infer_time, model_input_frame=None):
    global latest_waypoints, latest_base_velocity, latest_demo_frame
    global latest_demo_step, latest_demo_infer
    with demo_overlay_lock:
        wps = response.get("waypoints") if isinstance(response, dict) else None
        if wps is None and isinstance(response, dict):
            wps = response.get("trajectory")
        latest_waypoints = list(wps) if isinstance(wps, list) else []
        bv = response.get("base_velocity") if isinstance(response, dict) else None
        if isinstance(bv, list) and len(bv) >= 3:
            latest_base_velocity = [float(bv[0]), float(bv[1]), float(bv[2])]
        if model_input_frame is not None:
            latest_demo_frame = np.ascontiguousarray(model_input_frame.astype(np.uint8))
        latest_demo_step = int(step_idx)
        latest_demo_infer = float(infer_time)


def demo_video_render_thread():
    """Render waypoint-overlay video at fixed FPS."""
    frame_period = 1.0 / 30.0
    next_ts = time.time()
    while not exit_requested:
        if waypoint_recorder is None or not waypoint_recorder.running:
            time.sleep(0.05)
            continue

        rgb_for_log = None
        if manager is not None:
            rgb_depth_rw_lock.acquire_read()
            src = getattr(manager, 'raw_rgb_image', None)
            if src is None:
                src = manager.rgb_image
            if src is not None:
                rgb_for_log = copy.deepcopy(src)
            rgb_depth_rw_lock.release_read()

        if rgb_for_log is not None:
            with demo_overlay_lock:
                wps = list(latest_waypoints)
                bv = list(latest_base_velocity)
                step = latest_demo_step
                infer = latest_demo_infer
            overlay = draw_waypoint_overlay(rgb_for_log, wps, bv, step, infer)
            if overlay is not None:
                waypoint_recorder.update_frame(overlay)

        next_ts += frame_period
        sleep_time = next_ts - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_ts = time.time()


def paired_video_render_thread():
    """Record native RGB and the web-style 384x384 control overlay together."""
    fps = max(1, int(getattr(client_args, "record_video_fps", 15)))
    frame_period = 1.0 / float(fps)
    next_ts = time.time()
    while not exit_requested:
        raw_frame = None
        if manager is not None:
            rgb_depth_rw_lock.acquire_read()
            if getattr(manager, "raw_rgb_image", None) is not None:
                raw_frame = np.ascontiguousarray(manager.raw_rgb_image.copy())
            rgb_depth_rw_lock.release_read()

        with demo_overlay_lock:
            model_frame = None if latest_demo_frame is None else latest_demo_frame.copy()
            waypoints = list(latest_waypoints)
            base_velocity = list(latest_base_velocity)
            step = latest_demo_step
            infer = latest_demo_infer

        if raw_frame is not None and native_video_recorder is not None:
            native_video_recorder.update_frame(raw_frame)
        if model_frame is not None and control_video_recorder is not None:
            overlay = draw_waypoint_overlay(model_frame, waypoints, base_velocity, step, infer)
            if overlay is not None:
                control_video_recorder.update_frame(overlay)

        next_ts += frame_period
        sleep_time = next_ts - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_ts = time.time()


def print_latency_summary():
    with latency_samples_lock:
        samples = list(latency_samples)
    if not samples:
        print("[latency] no samples collected")
        return
    def stats(key):
        arr = sorted(float(s[key]) for s in samples if key in s and s[key] is not None)
        if not arr:
            return None
        n = len(arr)
        def pct(q):
            i = min(n - 1, max(0, int(q * n)))
            return arr[i]
        mean = sum(arr) / n
        return {"min": arr[0], "p50": pct(0.5), "p95": pct(0.95),
                "max": arr[-1], "mean": mean, "n": n}
    summary = {
        "transport": getattr(client_args, "transport", "http"),
        "camera_source": getattr(client_args, "camera_source", "realsense"),
        "video_network_interface": getattr(client_args, "video_network_interface", ""),
        "server_url": getattr(client_args, "server_url", ""),
        "tcp_url": getattr(client_args, "tcp_url", ""),
        "waypoint_strategy": _effective_waypoint_strategy(),
        "vx_positive_scale": float(getattr(client_args, "vx_positive_scale", 1.0)),
        "vx_negative_scale": float(getattr(client_args, "vx_negative_scale", 1.0)),
        "wz_scale": float(getattr(client_args, "wz_scale", 1.0)),
        "vx_positive_deadband": float(getattr(client_args, "vx_positive_deadband", 0.0)),
        "vx_negative_deadband": float(getattr(client_args, "vx_negative_deadband", 0.0)),
        "wz_deadband": float(getattr(client_args, "cmd_deadband_w", 0.0)),
        "max_vx": float(getattr(client_args, "cmd_max_v", 0.15)),
        "max_wz": float(getattr(client_args, "cmd_max_w", 0.30)),
        "vx_sign_hysteresis": float(getattr(client_args, "cmd_hysteresis_v", 0.0)),
        "ema_alpha": float(getattr(client_args, "cmd_ema_alpha", 1.0)),
        "yaw_boost": float(getattr(client_args, "yaw_boost", 1.0)),
        "yaw_boost_threshold": float(getattr(client_args, "yaw_boost_threshold", 0.10)),
        "yaw_boost_max": float(getattr(client_args, "yaw_boost_max", 1.5)),
        "http_ms": stats("http_ms"),
        "transport_overhead_ms": stats("transport_overhead_ms"),
        "network_upload_ms": stats("network_upload_ms"),
        "frame_age_ms": stats("frame_age_ms"),
        "cam_to_action_ms": stats("cam_to_action_ms"),
        "server_infer_ms": stats("server_infer_ms"),
        "server_request_ms": stats("server_request_ms"),
        "vision_encode_ms": stats("vision_encode_ms"),
        "vision_dino_ms": stats("vision_dino_ms"),
        "vision_siglip_ms": stats("vision_siglip_ms"),
        "vision_pool_ms": stats("vision_pool_ms"),
        "history_pack_ms": stats("history_pack_ms"),
        "policy_forward_ms": stats("policy_forward_ms"),
        "encode_ms": stats("encode_ms"),
        "action_v": stats("action_v"),
        "action_w": stats("action_w"),
        "server_action_v": stats("server_action_v"),
        "server_action_w": stats("server_action_w"),
        "waypoint_age_ms": stats("waypoint_age_ms"),
        "waypoint_latency_steps": stats("waypoint_latency_steps"),
        "waypoint_selected_index": stats("waypoint_selected_index"),
        "waypoint_selected_segment": stats("waypoint_selected_segment"),
        "waypoint_blend_alpha": stats("waypoint_blend_alpha"),
        "waypoint_plan_expired": stats("waypoint_plan_expired"),
        "waypoint_fused_vx_mean": stats("waypoint_fused_vx_mean"),
        "commanded_v": stats("commanded_v"),
        "commanded_w": stats("commanded_w"),
        "actual_v": stats("actual_v"),
        "actual_w": stats("actual_w"),
        "filter_clipped_v": stats("filter_clipped_v"),
        "filter_clipped_w": stats("filter_clipped_w"),
        "plan_age_ms": stats("plan_age_ms"),
        "plan_timeline_age_ms": stats("plan_timeline_age_ms"),
    }
    print("\n========== Latency summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=====================================")
    log_path = getattr(client_args, "latency_log_path", "") or ""
    log_alias = getattr(client_args, "latency_log_alias", "") or ""
    output_paths = []
    for path in (log_path, log_alias):
        if path and path not in output_paths:
            output_paths.append(path)
    payload = {"summary": summary, "samples": samples}
    for output_path in output_paths:
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            temp_path = f"{output_path}.tmp-{os.getpid()}"
            with open(temp_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, output_path)
            print(f"[latency] log -> {output_path}")
        except Exception as e:
            print(f"[latency] failed to write {output_path}: {e}")


def request_program_exit(reason):
    global exit_requested, video_recorder, waypoint_recorder
    if exit_requested:
        return
    exit_requested = True
    print(f"[exit] {reason}")
    if manager is not None:
        manager.move(0.0, 0.0, 0.0)
    if video_recorder is not None:
        video_recorder.stop()
    if waypoint_recorder is not None:
        waypoint_recorder.stop()
    if rclpy.ok():
        rclpy.shutdown()

# Shared memory for Unitree SDK communication
# odom_shm: [valid, px, py, pz, qw, qx, qy, qz, vx, vy, yaw_speed, recv_wall] = 12 doubles
# cmd_shm:  [flag, vx, vy, vyaw] = 4 doubles. flag=-1 requests StopMove.
odom_shm = None
cmd_shm = None


# ── Unitree SDK child process ─────────────────────────────────────────
def unitree_process(odom_shm, cmd_shm, network_interface):
    """Runs in a separate process with CycloneDDS — never imports rclpy."""
    import sys

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    ChannelFactoryInitialize(0, network_interface)

    def state_callback(msg: SportModeState_):
        odom_shm[1] = msg.position[0]
        odom_shm[2] = msg.position[1]
        odom_shm[3] = msg.position[2]
        q = msg.imu_state.quaternion  # [w, x, y, z]
        odom_shm[4] = q[0]
        odom_shm[5] = q[1]
        odom_shm[6] = q[2]
        odom_shm[7] = q[3]
        odom_shm[8] = msg.velocity[0]
        odom_shm[9] = msg.velocity[1]
        odom_shm[10] = msg.yaw_speed
        if len(odom_shm) > 11:
            odom_shm[11] = time.time()
        odom_shm[0] = 1.0  # mark valid

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(state_callback, 10)

    sport_client = SportClient()
    sport_client.SetTimeout(5.0)
    sport_client.Init()

    print(
        f"[unitree] SDK ready on {network_interface}, "
        "listening for SportModeState & cmd_vel",
        flush=True,
    )

    while True:
        if cmd_shm[0] < -0.5:
            sport_client.Move(0.0, 0.0, 0.0)
            time.sleep(0.05)
            code = sport_client.StopMove()
            cmd_shm[0] = 0.0
            print(f"[unitree] StopMove requested, code={code}", flush=True)
        elif cmd_shm[0] > 0.5:
            sport_client.Move(float(cmd_shm[1]), float(cmd_shm[2]), float(cmd_shm[3]))
            cmd_shm[0] = 0.0
        time.sleep(0.02)


# ── Odom polling thread (reads shared memory → fills manager) ─────────
def odom_poll_thread():
    """Runs in the main process, polls odom from shared memory at ~50 Hz."""
    global manager
    while manager is None and not exit_requested:
        time.sleep(0.1)

    if exit_requested:
        return

    odom_cnt = 0
    while not exit_requested:
        if odom_shm[0] > 0.5:
            qz = odom_shm[7]
            qw = odom_shm[4]
            yaw = math.atan2(2 * qz * qw, 1 - 2 * qz * qz)
            px, py = odom_shm[1], odom_shm[2]
            vx = odom_shm[8]
            yaw_speed = odom_shm[10]

            odom_cnt += 1
            odom_rw_lock.acquire_write()
            manager.odom = [px, py, yaw]
            manager.odom_queue.append((time.time(), copy.deepcopy(manager.odom)))
            manager.odom_timestamp = time.time()
            manager.linear_vel = vx
            manager.angular_vel = yaw_speed
            odom_rw_lock.release_write()

            R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            manager.homo_odom = np.eye(4)
            manager.homo_odom[:2, :2] = R0
            manager.homo_odom[:2, 3] = [px, py]
            manager.vel = [vx, yaw_speed]
            manager.odom_cnt = odom_cnt

            if odom_cnt == 1:
                manager.homo_goal = manager.homo_odom.copy()
                print(f"[odom] First odom received: x={px:.3f} y={py:.3f} yaw={yaw:.3f}")

        time.sleep(0.02)


def dual_sys_eval(image_bytes, rgb_stamp, front_image_bytes, url='http://127.0.0.1:5801/eval_dual'):
    global policy_init, http_idx, first_running_time, http_session, tcp_eval_client
    data = {"reset": policy_init, "idx": http_idx}
    data['instruction'] = client_args.instruction
    with manager.intrinsic_lock:
        intrinsic = manager.camera_intrinsic
    if intrinsic is not None:
        data['camera_intrinsic'] = intrinsic.tolist()
    data['client_control_mode'] = str(getattr(client_args, 'action_source', 'server_velocity'))
    data['client_exec_velocity'] = [float(desired_v), 0.0, float(desired_w)]

    policy_init = False
    start = time.time()
    data['client_send_timestamp'] = start
    transport = getattr(client_args, "transport", "http")
    if transport in ("tcp_jpeg", "tcp_raw"):
        if transport == "tcp_raw":
            payload_bytes = image_bytes if isinstance(image_bytes, (bytes, bytearray, memoryview)) else image_bytes.getvalue()
            rgb_depth_rw_lock.acquire_read()
            upload_shape = tuple(getattr(manager, "last_upload_shape", ()) or ())
            rgb_depth_rw_lock.release_read()
            if len(upload_shape) >= 3:
                data["image_height"] = int(upload_shape[0])
                data["image_width"] = int(upload_shape[1])
                data["image_channels"] = int(upload_shape[2])
            data["image_encoding"] = "raw_rgb"
        else:
            payload_bytes = image_bytes.getvalue()
            data["image_encoding"] = "jpeg"
        out = tcp_eval_client.request(data, payload_bytes)
    else:
        json_data = json.dumps(data)
        files = {
            'image': ('rgb_image', image_bytes, 'image/jpeg'),
        }
        response = http_session.post(url, files=files, data={'json': json_data}, timeout=30)
        out = json.loads(response.text)
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    out['_client_http_roundtrip_ms'] = (time.time() - start) * 1000.0
    out['_frame_age_ms'] = max(0.0, (start - rgb_stamp) * 1000.0)
    out['_client_send_wall'] = start
    return out


def cmd_dispatch_thread():
    """50 Hz: roll current_plan into a (v, w) and send via manager.move().

    Single writer to manager.move() across the whole client. Replaces the
    previous pattern where planning_thread (1.4 Hz) and control_thread
    (10 Hz) raced on cmd_shm.
    """
    global last_dispatch_time, last_inter_update_ms, last_commanded_v, last_commanded_w
    period = 1.0 / DISPATCH_HZ
    next_tick = time.time()
    dispatch_count = 0
    while True:
        if exit_requested:
            if manager is not None:
                manager.move(0.0, 0.0, 0.0)
            last_commanded_v, last_commanded_w = 0.0, 0.0
            return

        with plan_lock:
            kind = current_plan["kind"]
            opt_u = current_plan["opt_u"]
            arrival_time = current_plan["arrival_time"]
            timeline_start = current_plan["timeline_start"]
            dt = current_plan["dt"]
            horizon = current_plan["horizon"]
            plan_version = current_plan["version"]

        v, w = 0.0, 0.0
        selected_segment = -1
        plan_expired = False
        if kind is not None and opt_u is not None and arrival_time > 0.0:
            elapsed = time.time() - arrival_time
            if elapsed > MAX_PLAN_AGE_S:
                # Safety: plan too stale, brake to 0
                v, w = 0.0, 0.0
                plan_expired = True
            elif kind == "two_step":
                v, w, selected_segment = two_step_velocity_at_elapsed(opt_u, elapsed, dt)
            else:  # "server_velocity" - single (v, w)
                v = float(opt_u[0])
                w = float(opt_u[1])

        planned_v, planned_w = v, w
        # ── Velocity scale + low-speed filters (all default = pass-through) ──
        global _ema_v, _ema_w, _prev_sign_v, last_raw_v, last_raw_w
        if client_args is not None:
            v, w = scale_velocity_command(
                v,
                w,
                vx_positive_scale=float(getattr(client_args, "vx_positive_scale", 1.0)),
                vx_negative_scale=float(getattr(client_args, "vx_negative_scale", 1.0)),
                wz_scale=float(getattr(client_args, "wz_scale", 1.0)),
            )
        raw_v, raw_w = v, w
        last_raw_v, last_raw_w = raw_v, raw_w
        if client_args is not None:
            # 0. yaw boost: amplify model's turn intent when it's meaningful.
            # Goal: catch fast-moving targets that would otherwise drift out of FOV.
            yaw_boost = float(getattr(client_args, "yaw_boost", 1.0))
            yaw_thresh = float(getattr(client_args, "yaw_boost_threshold", 0.10))
            yaw_max = float(getattr(client_args, "yaw_boost_max", 1.5))
            if yaw_boost != 1.0 and abs(w) > yaw_thresh:
                w = max(-yaw_max, min(yaw_max, w * yaw_boost))
            # 1. dead-band: zero out tiny commands (root cause of low-speed jitter)
            dz_w = float(getattr(client_args, "cmd_deadband_w", 0.0))
            v = apply_directional_vx_deadband(
                v,
                positive_deadband=float(getattr(client_args, "vx_positive_deadband", 0.0)),
                negative_deadband=float(getattr(client_args, "vx_negative_deadband", 0.0)),
            )
            if dz_w > 0.0 and abs(w) < dz_w:
                w = 0.0
            # 2. sign-flip hysteresis on vx: ignore brief sign reversals near zero
            hyst_v = float(getattr(client_args, "cmd_hysteresis_v", 0.0))
            if hyst_v > 0.0:
                sign = 1 if v > hyst_v else (-1 if v < -hyst_v else 0)
                if sign == 0:
                    sign = _prev_sign_v  # keep previous sign in the dead zone
                _prev_sign_v = sign
                if sign == 0:
                    v = 0.0
                elif (sign > 0 and v < 0.0) or (sign < 0 and v > 0.0):
                    v = 0.0  # transient flip suppressed until amplitude exceeds hyst
            # 3. EMA smoothing (only useful when alpha < 1.0)
            alpha = float(getattr(client_args, "cmd_ema_alpha", 1.0))
            if 0.0 < alpha < 1.0:
                _ema_v = alpha * v + (1.0 - alpha) * _ema_v
                _ema_w = alpha * w + (1.0 - alpha) * _ema_w
                v, w = _ema_v, _ema_w

            # Final client-side hard limits remain effective even if the
            # server emits a larger velocity than expected.
            max_v = max(0.0, float(getattr(client_args, "cmd_max_v", 0.15)))
            max_w = max(0.0, float(getattr(client_args, "cmd_max_w", 0.30)))
            v = max(-max_v, min(max_v, v)) if max_v > 0.0 else 0.0
            w = max(-max_w, min(max_w, w)) if max_w > 0.0 else 0.0

        if manager is not None:
            now = time.time()
            if last_dispatch_time > 0:
                last_inter_update_ms = (now - last_dispatch_time) * 1000.0
            last_dispatch_time = now
            last_commanded_v, last_commanded_w = v, w
            manager.move(v, 0.0, w)

            dispatch_count += 1
            if getattr(client_args, "diagnostic_log_path", "") and dispatch_count % 5 == 0:
                rgb_recv_wall = float(getattr(manager, "rgb_recv_wall", 0.0) or 0.0)
                odom_recv_wall = (
                    float(odom_shm[11])
                    if odom_shm is not None and len(odom_shm) > 11
                    else 0.0
                )
                queue_diagnostic_event(
                    "dispatch",
                    plan_kind=str(kind),
                    plan_version=int(plan_version),
                    plan_age_ms=(now - arrival_time) * 1000.0 if arrival_time > 0.0 else None,
                    plan_timeline_age_ms=(now - timeline_start) * 1000.0 if timeline_start > 0.0 else None,
                    plan_expired=bool(plan_expired),
                    selected_segment=int(selected_segment),
                    planned_v=float(planned_v),
                    planned_w=float(planned_w),
                    raw_v=float(raw_v),
                    raw_w=float(raw_w),
                    commanded_v=float(v),
                    commanded_w=float(w),
                    actual_v=float(getattr(manager, "linear_vel", 0.0)),
                    actual_w=float(getattr(manager, "angular_vel", 0.0)),
                    odom_age_ms=(now - odom_recv_wall) * 1000.0 if odom_recv_wall > 0.0 else None,
                    camera_age_ms=(now - rgb_recv_wall) * 1000.0 if rgb_recv_wall > 0.0 else None,
                    frame_seq=int(getattr(manager, "frame_seq", 0)),
                )

        next_tick += period
        slack = next_tick - time.time()
        if slack > 0:
            time.sleep(slack)
        else:
            # Behind schedule (e.g. paused for GC); resync without busy loop
            next_tick = time.time()


def planning_thread():
    global exit_requested
    last_sent_frame_seq = 0

    while True:
        if exit_requested:
            return

        if not manager.new_image_arrived:
            time.sleep(0.002)
            continue
        manager.new_image_arrived = False
        manager.wait_for_fresh_frame(
            last_sent_frame_seq,
            float(client_args.max_wait_new_frame_ms) / 1000.0,
            float(client_args.fresh_frame_age_threshold_ms) / 1000.0,
        )
        # JIT encode the freshest raw frame right before HTTP send. This eliminates
        # the staleness that comes from encoding at camera rate (~33ms per frame)
        # while only sending at ~2.5 Hz — the JPEG bytes are always built from the
        # newest cv_bridge-decoded image, not one that may be 30+ms older.
        rgb_bytes, rgb_time, rgb_recv_wall, frame_seq, _enc_ms = manager.encode_latest_rgb()
        if rgb_bytes is None:
            time.sleep(0.005)
            continue
        last_sent_frame_seq = int(frame_seq)

        odom_infer = None
        if rgb_bytes is not None:
            step_t0 = time.time()
            response = dual_sys_eval(rgb_bytes, rgb_time, None, url=client_args.server_url)

            latency_metrics = {
                "frame_seq": int(frame_seq),
                "frame_age_ms": float(response.get('_frame_age_ms', 0.0)),
                "callback_lag_ms": max(0.0, (rgb_recv_wall - rgb_time) * 1000.0),
                "queue_wait_ms": max(0.0, (float(response.get('_client_send_wall', time.time())) - rgb_recv_wall) * 1000.0),
                "encode_ms": float(manager.last_encode_ms),
                "client_http_roundtrip_ms": float(response.get('_client_http_roundtrip_ms', 0.0)),
                "network_upload_ms": float(response.get('network_upload_latency', 0.0) or 0.0) * 1000.0,
                "server_request_ms": float(response.get('request_time', 0.0) or 0.0) * 1000.0,
                "server_infer_ms": float(response.get('infer_time', 0.0) or 0.0) * 1000.0,
                "server_noninfer_ms": float(response.get('noninfer_delay', 0.0) or 0.0) * 1000.0,
                "overall_server_ready_ms": float(response.get('overall_latency', 0.0) or 0.0) * 1000.0,
                "transport_overhead_ms": max(
                    0.0,
                    float(response.get('_client_http_roundtrip_ms', 0.0))
                    - float(response.get('request_time', 0.0) or 0.0) * 1000.0,
                ),
                "post_http_to_action_ms": 0.0,
                "camera_to_action_ms": 0.0,
                "action_source": client_args.action_source,
            }
            if 'base_velocity' in response:
                base_velocity = response.get('base_velocity', [0.0, 0.0, 0.0])
                server_v = float(base_velocity[0]) if len(base_velocity) > 0 else 0.0
                server_w = float(base_velocity[2]) if len(base_velocity) > 2 else 0.0
                v, w, waypoint_info, waypoint_plan = _select_server_waypoint_action(
                    response,
                    base_velocity,
                    rgb_time,
                    odom_infer,
                )
                desired_v, desired_w = v, w
                if waypoint_plan is None:
                    _publish_plan("server_velocity", np.array([v, w], dtype=np.float64))
                else:
                    _publish_plan(**waypoint_plan)
                action_t1 = time.time()
                latency_metrics["post_http_to_action_ms"] = (action_t1 - (step_t0 + response.get('_client_http_roundtrip_ms', 0.0) / 1000.0)) * 1000.0
                latency_metrics["camera_to_action_ms"] = (action_t1 - rgb_time) * 1000.0
                latency_metrics["action_v"] = v
                latency_metrics["action_w"] = w
                latency_metrics["server_action_v"] = server_v
                latency_metrics["server_action_w"] = server_w
                latency_metrics.update(waypoint_info)
            else:
                _publish_plan("server_velocity", np.zeros(2, dtype=np.float64))
                print("[control] response has no base_velocity; publishing zero command")
            infer_time = time.time() - step_t0
            timing = response.get("timing", {}) if isinstance(response.get("timing", {}), dict) else {}
            # Step 1: extra fields for waypoint→velocity diagnosis.
            with plan_lock:
                plan_arrival = current_plan["arrival_time"]
                plan_timeline_start = current_plan["timeline_start"]
                plan_kind = current_plan["kind"]
                plan_version = current_plan["version"]
            actual_v_now = float(manager.linear_vel) if manager is not None else 0.0
            actual_w_now = float(manager.angular_vel) if manager is not None else 0.0
            latency_metrics["dispatch_inter_update_ms"] = float(last_inter_update_ms)
            latency_metrics["plan_age_ms"] = (time.time() - plan_arrival) * 1000.0 if plan_arrival > 0 else 0.0
            latency_metrics["plan_timeline_age_ms"] = (
                (time.time() - plan_timeline_start) * 1000.0
                if plan_timeline_start > 0 else 0.0
            )
            latency_metrics["plan_kind"] = str(plan_kind)
            latency_metrics["plan_version"] = int(plan_version)
            latency_metrics["commanded_v"] = float(last_commanded_v)
            latency_metrics["commanded_w"] = float(last_commanded_w)
            latency_metrics["raw_cmd_v"] = float(last_raw_v)
            latency_metrics["raw_cmd_w"] = float(last_raw_w)
            latency_metrics["filter_clipped_v"] = float(last_raw_v - last_commanded_v)
            latency_metrics["filter_clipped_w"] = float(last_raw_w - last_commanded_w)
            # If yaw-boost > 1, this is positive when boost was applied (commanded > raw).
            latency_metrics["yaw_boost_delta"] = float(last_commanded_w - last_raw_w)
            latency_metrics["actual_v"] = actual_v_now
            latency_metrics["actual_w"] = actual_w_now
            latency_metrics["vel_track_err_v"] = float(last_commanded_v - actual_v_now)
            latency_metrics["vel_track_err_w"] = float(last_commanded_w - actual_w_now)
            update_video_overlay(response, http_idx, infer_time, latency_metrics=latency_metrics)
            model_input_frame = None
            if getattr(client_args, "record_video_pair", False) and manager is not None:
                rgb_depth_rw_lock.acquire_read()
                if manager.rgb_image is not None:
                    model_input_frame = manager.rgb_image.copy()
                rgb_depth_rw_lock.release_read()
            update_demo_overlay(
                response,
                http_idx,
                infer_time,
                model_input_frame=model_input_frame,
            )
            sample = {
                    "t": time.time(),
                    "http_ms": float(latency_metrics["client_http_roundtrip_ms"]),
                    "transport_overhead_ms": float(latency_metrics["transport_overhead_ms"]),
                    "network_upload_ms": float(latency_metrics["network_upload_ms"]),
                    "frame_age_ms": float(latency_metrics["frame_age_ms"]),
                    "cam_to_action_ms": float(latency_metrics["camera_to_action_ms"]),
                    "server_infer_ms": float(latency_metrics["server_infer_ms"]),
                    "server_request_ms": float(latency_metrics["server_request_ms"]),
                    "vision_encode_ms": float(response.get("vision_encode_ms", 0.0) or timing.get("vision_encode_ms", 0.0) or 0.0),
                    "vision_dino_ms": float(response.get("vision_dino_ms", 0.0) or timing.get("vision_dino_ms", 0.0) or 0.0),
                    "vision_siglip_ms": float(response.get("vision_siglip_ms", 0.0) or timing.get("vision_siglip_ms", 0.0) or 0.0),
                    "vision_pool_ms": float(response.get("vision_pool_ms", 0.0) or timing.get("vision_pool_ms", 0.0) or 0.0),
                    "history_pack_ms": float(response.get("history_pack_ms", 0.0) or timing.get("history_pack_ms", 0.0) or 0.0),
                    "policy_forward_ms": float(response.get("policy_forward_ms", 0.0) or timing.get("policy_forward_ms", 0.0) or 0.0),
                    "encode_ms": float(latency_metrics["encode_ms"]),
                    "action_v": float(latency_metrics.get("action_v", 0.0)),
                    "action_w": float(latency_metrics.get("action_w", 0.0)),
                    "server_action_v": float(latency_metrics.get("server_action_v", latency_metrics.get("action_v", 0.0))),
                    "server_action_w": float(latency_metrics.get("server_action_w", latency_metrics.get("action_w", 0.0))),
                    "waypoint_strategy": str(latency_metrics.get("waypoint_strategy", _effective_waypoint_strategy())),
                    "waypoint_selection_source": str(latency_metrics.get("waypoint_selection_source", "unknown")),
                    "waypoint_count": int(latency_metrics.get("waypoint_count", 0)),
                    "waypoint_age_ms": float(latency_metrics.get("waypoint_age_ms", 0.0)),
                    "waypoint_latency_steps": float(latency_metrics.get("waypoint_latency_steps", 0.0)),
                    "waypoint_selected_index": int(latency_metrics.get("waypoint_selected_index", -1)),
                    "waypoint_selected_segment": int(latency_metrics.get("waypoint_selected_segment", -1)),
                    "waypoint_blend_alpha": float(latency_metrics.get("waypoint_blend_alpha", 0.0)),
                    "waypoint_plan_expired": float(latency_metrics.get("waypoint_plan_expired", 0.0)),
                    "waypoint_selected_x": float(latency_metrics.get("waypoint_selected_x", 0.0)),
                    "waypoint_selected_y": float(latency_metrics.get("waypoint_selected_y", 0.0)),
                    "waypoint_selected_yaw": float(latency_metrics.get("waypoint_selected_yaw", 0.0)),
                    "waypoint_fused_indices": list(latency_metrics.get("waypoint_fused_indices", [])),
                    "waypoint_fused_vx": list(latency_metrics.get("waypoint_fused_vx", [])),
                    "waypoint_fused_vx_mean": float(latency_metrics.get("waypoint_fused_vx_mean", 0.0)),
                    "waypoint_yaw_index": int(latency_metrics.get("waypoint_yaw_index", -1)),
                    "raw_cmd_v": float(latency_metrics.get("raw_cmd_v", 0.0)),
                    "raw_cmd_w": float(latency_metrics.get("raw_cmd_w", 0.0)),
                    "commanded_v": float(latency_metrics.get("commanded_v", 0.0)),
                    "commanded_w": float(latency_metrics.get("commanded_w", 0.0)),
                    "actual_v": float(latency_metrics.get("actual_v", 0.0)),
                    "actual_w": float(latency_metrics.get("actual_w", 0.0)),
                    "filter_clipped_v": float(latency_metrics.get("filter_clipped_v", 0.0)),
                    "filter_clipped_w": float(latency_metrics.get("filter_clipped_w", 0.0)),
                    "plan_age_ms": float(latency_metrics.get("plan_age_ms", 0.0)),
                    "plan_timeline_age_ms": float(latency_metrics.get("plan_timeline_age_ms", 0.0)),
                }
            with latency_samples_lock:
                latency_samples.append(sample)
            if client_args.diagnostic_log_path:
                queue_diagnostic_event("inference", **sample)
            print_every = int(getattr(client_args, "latency_print_every", 0))
            if print_every > 0 and int(latency_metrics["plan_version"]) % print_every == 0:
                print(json.dumps({"latency": latency_metrics, "timing": response.get("timing", {})}, ensure_ascii=False))
        else:
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None}"
            )
            time.sleep(0.01)


class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')

        self.camera_source = str(getattr(client_args, "camera_source", "realsense"))

        # depth=1: DDS layer keeps only the freshest frame. With server_infer ~350ms
        # and camera 30Hz, a depth=10 buffer adds ~150-200ms of stale-frame backlog
        # before rclpy spin drains it. We always want the latest frame, never a queued one.
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.camera_info_sub = None
        if self.camera_source == "realsense":
            self.create_subscription(Image, "/camera/color/image_raw", self._rgb_callback, qos_profile)
            if getattr(client_args, "send_depth", False):
                self.create_subscription(
                    Image, "/camera/aligned_depth_to_color/image_raw", self._depth_callback, qos_profile
                )
            self.camera_info_sub = self.create_subscription(
                CameraInfo, "/camera/color/camera_info", self.camera_info_callback, qos_profile
            )

        # camera intrinsic (populated by CameraInfo callback)
        self.intrinsic_lock = threading.Lock()
        self.camera_intrinsic = None

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.raw_rgb_image = None
        self.raw_rgb_jpeg = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0
        self.rgb_recv_wall = 0.0
        self.frame_seq = 0
        self.last_encode_ms = 0.0
        self.last_upload_shape = None
        self.last_crop_geometry = None

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

        self.video_client = None
        self.capture_thread = None
        self.camera_preview_thread = None
        if self.camera_source == "go2":
            self._start_video_client()
        self._start_camera_preview()

    @staticmethod
    def _stamp_to_sec(stamp):
        """Convert header stamp to float seconds, using abs() to handle negative hardware timestamps."""
        return abs(stamp.sec) + stamp.nanosec / 1.0e9

    def _rgb_callback(self, msg):
        self.rgb_only_callback(msg)

    def _depth_callback(self, msg):
        if getattr(client_args, "send_depth", False):
            raw_depth = self.cv_bridge.imgmsg_to_cv2(msg, '16UC1')
            raw_depth[np.isnan(raw_depth)] = 0
            raw_depth[np.isinf(raw_depth)] = 0
            self.depth_image = raw_depth / 1000.0

    def camera_info_callback(self, msg):
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        intrinsic = np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        with self.intrinsic_lock:
            self.camera_intrinsic = intrinsic

    def rgb_forward_callback(self, rgb_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True
        with rgb_frame_cv:
            rgb_frame_cv.notify_all()

    def rgb_only_callback(self, rgb_msg):
        # Lightweight callback: only cv_bridge convert + stash raw frame + stamp.
        # JPEG resize/encode is moved to encode_latest_rgb() called by planning_thread
        # just before HTTP send, so the encoded frame is always the freshest possible
        # one and we don't burn CPU encoding 30 frames/sec when we only send ~2.5/sec.
        if getattr(client_args, "rgb_decode", "fast") == "fast" and str(getattr(rgb_msg, "encoding", "")).lower() == "rgb8":
            raw_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
        else:
            raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        recv_wall = time.time()
        stamp_sec = self._stamp_to_sec(rgb_msg.header.stamp)

        rgb_depth_rw_lock.acquire_write()
        self.raw_rgb_image = raw_image
        self.raw_rgb_jpeg = None
        self.rgb_time = stamp_sec
        self.rgb_recv_wall = recv_wall
        self.last_rgb_time = stamp_sec
        self.frame_seq += 1
        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True
        with rgb_frame_cv:
            rgb_frame_cv.notify_all()

    def _start_video_client(self):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.video.video_client import VideoClient
        except ImportError as exc:
            raise RuntimeError("Go2 camera mode requires unitree_sdk2py with VideoClient") from exc

        interface = str(getattr(client_args, "video_network_interface", "") or "").strip()
        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            ChannelFactoryInitialize(0)
        self.video_client = VideoClient()
        self.video_client.SetTimeout(float(getattr(client_args, "video_client_timeout", 3.0)))
        self.video_client.Init()
        with self.intrinsic_lock:
            self.camera_intrinsic = np.eye(4, dtype=np.float64)
        self.capture_thread = threading.Thread(
            target=self._video_client_capture_loop,
            name="go2-video-client",
            daemon=True,
        )
        self.capture_thread.start()
        print(
            f"[camera] source=go2 VideoClient interface={interface or '<default>'} "
            f"timeout={float(getattr(client_args, 'video_client_timeout', 3.0)):.1f}s"
        )

    def _start_camera_preview(self):
        preview_fps = float(getattr(client_args, "camera_preview_fps", 0.0))
        if preview_fps > 0.0:
            self.camera_preview_thread = threading.Thread(
                target=self._camera_preview_loop,
                name="camera-preview",
                daemon=True,
            )
            self.camera_preview_thread.start()

    def _video_client_capture_loop(self):
        last_error_code = None
        while not exit_requested:
            try:
                code, data = self.video_client.GetImageSample()
            except Exception as exc:
                if last_error_code != "exception":
                    print(f"[video_client] GetImageSample exception: {exc}")
                    last_error_code = "exception"
                time.sleep(0.1)
                continue
            if code != 0:
                if code != last_error_code:
                    print(f"[video_client] GetImageSample error code: {code}")
                    last_error_code = code
                time.sleep(0.05)
                continue
            last_error_code = None
            self.store_video_client_frame(bytes(data), recv_wall=time.time())

    def _camera_preview_loop(self):
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(str(client_args.server_url))
        preview_url = urlunparse(parsed._replace(path="/api/camera-frame", params="", query="", fragment=""))
        period = 1.0 / max(0.1, float(client_args.camera_preview_fps))
        session = requests.Session()
        session.trust_env = False
        last_frame_seq = -1
        last_error = None
        next_send = time.monotonic()
        print(f"[camera-preview] publishing raw camera JPEG to {preview_url} at {1.0 / period:.1f} FPS")
        try:
            while not exit_requested:
                now = time.monotonic()
                if now < next_send:
                    time.sleep(min(0.02, next_send - now))
                    continue
                next_send = now + period

                rgb_depth_rw_lock.acquire_read()
                source_jpeg = self.raw_rgb_jpeg
                raw_rgb = self.raw_rgb_image
                frame_seq = self.frame_seq
                rgb_depth_rw_lock.release_read()
                if raw_rgb is None or frame_seq == last_frame_seq:
                    continue
                jpeg_bytes = encode_camera_preview_jpeg(
                    raw_rgb,
                    source_jpeg=source_jpeg,
                    quality=int(getattr(client_args, "jpeg_quality", 60)),
                )
                if jpeg_bytes is None:
                    continue
                try:
                    response = session.post(
                        preview_url,
                        data=jpeg_bytes,
                        headers={"Content-Type": "image/jpeg", "X-Frame-Seq": str(frame_seq)},
                        timeout=1.0,
                    )
                    response.raise_for_status()
                    last_frame_seq = frame_seq
                    last_error = None
                except Exception as exc:
                    error_name = type(exc).__name__
                    if error_name != last_error:
                        print(f"[camera-preview] publish failed: {error_name}: {exc}")
                        last_error = error_name
                    time.sleep(0.25)
        finally:
            session.close()

    def stop_video_client(self):
        """Stop VideoClient workers before CycloneDDS is torn down."""
        if self.camera_preview_thread is not None and self.camera_preview_thread.is_alive():
            self.camera_preview_thread.join(timeout=2.0)
            if self.camera_preview_thread.is_alive():
                print("[camera-preview] WARNING: preview thread did not stop before timeout")
        if self.capture_thread is not None and self.capture_thread.is_alive():
            timeout = float(getattr(client_args, "video_client_timeout", 3.0)) + 1.0
            self.capture_thread.join(timeout=timeout)
            if self.capture_thread.is_alive():
                print("[camera] WARNING: capture thread did not stop before timeout")

        # unitree_sdk2py 1.x has no public Client.close().  Close the channels
        # explicitly so its response-reader thread is joined before Python
        # unloads CycloneDDS.  Leaving that C++/Python thread alive can make the
        # next launch fail in Domain initialization.
        if self.video_client is not None:
            stub = getattr(self.video_client, "_ClientBase__stub", None)
            channels = (
                ("_ClientStub__recvChannel", "CloseReader"),
                ("_ClientStub__sendChannel", "CloseWriter"),
            )
            for attribute, close_method in channels:
                channel = getattr(stub, attribute, None) if stub is not None else None
                if channel is None:
                    continue
                try:
                    getattr(channel, close_method)()
                except Exception as exc:
                    print(f"[camera] WARNING: {close_method} failed: {exc}")
            self.video_client = None

    def store_video_client_frame(self, jpeg_bytes, recv_wall=None):
        """Decode and publish one VideoClient JPEG into the common latest-frame path."""
        if not jpeg_bytes:
            return False
        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3:
            return False
        raw_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        recv_wall = time.time() if recv_wall is None else float(recv_wall)

        rgb_depth_rw_lock.acquire_write()
        self.raw_rgb_image = raw_image
        self.raw_rgb_jpeg = jpeg_bytes
        self.rgb_time = recv_wall
        self.rgb_recv_wall = recv_wall
        self.last_rgb_time = recv_wall
        self.frame_seq += 1
        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True
        with rgb_frame_cv:
            rgb_frame_cv.notify_all()
        return True

    def encode_latest_rgb(self):
        """JIT encode the freshest raw frame to JPEG. Called by planning_thread
        right before each HTTP send. Returns (rgb_bytes, rgb_time, rgb_recv_wall,
        frame_seq, encode_ms) or (None, ...) if no frame yet.
        """
        rgb_depth_rw_lock.acquire_read()
        raw = self.raw_rgb_image
        source_jpeg = self.raw_rgb_jpeg
        rgb_time = self.rgb_time
        rgb_recv_wall = self.rgb_recv_wall
        frame_seq = self.frame_seq
        rgb_depth_rw_lock.release_read()

        if raw is None:
            return None, 0.0, 0.0, 0, 0.0

        t0 = time.time()
        upload_image = raw
        input_mode = str(getattr(client_args, "model_input_mode", "aspect_resize")).strip().lower()
        if input_mode in ("center_crop_height", "center_crop_720p"):
            upload_image, crop_geometry = prepare_center_crop_height(
                raw,
                crop_size=int(client_args.model_crop_size),
            )
            if crop_geometry != self.last_crop_geometry:
                target_w, target_h, x0, y0 = crop_geometry
                print(
                    f"[camera] model input: {raw.shape[1]}x{raw.shape[0]} -> "
                    f"{target_w}x{target_h} -> center crop "
                    f"x={x0}, y={y0}, size={upload_image.shape[1]}x{upload_image.shape[0]}"
                )
                self.last_crop_geometry = crop_geometry
        elif int(client_args.upload_width) > 0:
            src_h, src_w = raw.shape[:2]
            dst_w = int(client_args.upload_width)
            if int(client_args.upload_height) > 0:
                dst_h = int(client_args.upload_height)
            else:
                dst_h = max(1, int(round(src_h * (dst_w / float(src_w)))))
            if dst_w != src_w or dst_h != src_h:
                upload_image = cv2.resize(raw, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        resized = upload_image is not raw
        if getattr(client_args, "transport", "http") == "tcp_raw":
            image_bytes = np.ascontiguousarray(upload_image, dtype=np.uint8).tobytes()
        elif source_jpeg is not None and not resized:
            image_bytes = io.BytesIO(source_jpeg)
        elif getattr(client_args, "jpeg_encoder", "opencv") == "opencv":
            ok, enc = cv2.imencode(
                ".jpg",
                cv2.cvtColor(upload_image, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(client_args.jpeg_quality)],
            )
            if not ok:
                return None, 0.0, 0.0, 0, 0.0
            image_bytes = io.BytesIO(enc.tobytes())
        else:
            image = PIL_Image.fromarray(upload_image)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG', quality=int(client_args.jpeg_quality))
            image_bytes.seek(0)
        encode_ms = (time.time() - t0) * 1000.0

        rgb_depth_rw_lock.acquire_write()
        self.rgb_image = upload_image
        self.rgb_bytes = image_bytes
        self.last_encode_ms = encode_ms
        self.last_upload_shape = tuple(upload_image.shape)
        rgb_depth_rw_lock.release_write()

        return image_bytes, rgb_time, rgb_recv_wall, frame_seq, encode_ms

    def wait_for_fresh_frame(self, last_frame_seq, max_wait_s, stale_age_s):
        if max_wait_s <= 0.0:
            return False
        rgb_depth_rw_lock.acquire_read()
        frame_seq = self.frame_seq
        rgb_recv_wall = self.rgb_recv_wall
        rgb_depth_rw_lock.release_read()
        if frame_seq > int(last_frame_seq) and (time.time() - rgb_recv_wall) <= stale_age_s:
            return True
        target_seq = frame_seq
        deadline = time.time() + max_wait_s
        while time.time() < deadline and not exit_requested:
            remaining = max(0.0, deadline - time.time())
            with rgb_frame_cv:
                rgb_frame_cv.wait(timeout=min(0.005, remaining))
            rgb_depth_rw_lock.acquire_read()
            frame_seq = self.frame_seq
            rgb_recv_wall = self.rgb_recv_wall
            rgb_depth_rw_lock.release_read()
            if frame_seq > int(target_seq):
                return True
            if (
                frame_seq > int(last_frame_seq)
                and stale_age_s > 0.0
                and (time.time() - rgb_recv_wall) <= stale_age_s
            ):
                return True
        return False

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        if getattr(client_args, "dry_run", False):
            self.last_cmd = [float(vx), float(vy), float(vyaw)]
            return
        # Send move command to Unitree SDK via shared memory
        cmd_shm[1] = vx
        cmd_shm[2] = vy
        cmd_shm[3] = vyaw
        cmd_shm[0] = 1.0  # signal new command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, required=True, help="Navigation instruction for the agent")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5801/eval_dual", help="DualVLN server URL")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["http", "tcp_jpeg", "tcp_raw"],
        help="Inference transport. tcp_jpeg/tcp_raw use the persistent TCP protocol.",
    )
    parser.add_argument(
        "--tcp_url",
        type=str,
        default="",
        help="TCP inference URL, e.g. tcp://10.129.32.220:5803. Empty derives host from --server_url and port 5803.",
    )
    parser.add_argument(
        "--action_source",
        type=str,
        default="server_velocity",
        choices=["server_velocity"],
        help="Use the server waypoint/velocity response.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=60, help="JPEG quality for RGB upload.")
    parser.add_argument(
        "--camera_source",
        type=str,
        default="realsense",
        choices=["realsense", "go2"],
        help="RGB source: external ROS RealSense or Go2 front camera via SDK VideoClient.",
    )
    parser.add_argument(
        "--video_network_interface",
        type=str,
        default="",
        help="Network interface passed to Unitree ChannelFactoryInitialize in Go2 camera mode.",
    )
    parser.add_argument(
        "--video_client_timeout",
        type=float,
        default=3.0,
        help="VideoClient GetImageSample RPC timeout in seconds.",
    )
    parser.add_argument("--send_depth", action="store_true", help="Upload depth too. Disabled by default.")
    parser.add_argument("--dry_run", action="store_true", help="Do not send velocity to the robot.")
    parser.add_argument(
        "--enable-live-control",
        action="store_true",
        help="Required safety acknowledgement when --dry_run is not used.",
    )
    parser.add_argument("--upload_width", type=int, default=384, help="Resize RGB before upload. <=0 disables.")
    parser.add_argument("--upload_height", type=int, default=0, help="Optional upload height. 0 keeps aspect ratio.")
    parser.add_argument(
        "--model-input-mode",
        choices=["aspect_resize", "center_crop_height", "center_crop_720p"],
        default="aspect_resize",
        help="Model input preprocessing. center_crop_height resizes height to the crop size, then center-crops.",
    )
    parser.add_argument("--model-downsample-height", type=int, default=720)
    parser.add_argument("--model-crop-size", type=int, default=384)
    parser.add_argument(
        "--camera-preview-fps",
        type=float,
        default=0.0,
        help="Publish native Go2 JPEG frames to the local web server at this rate. 0 disables.",
    )
    parser.add_argument(
        "--jpeg_encoder",
        type=str,
        default="opencv",
        choices=["opencv", "pil"],
        help="JPEG encoder for RGB upload. OpenCV is usually faster on GO2.",
    )
    parser.add_argument(
        "--rgb_decode",
        type=str,
        default="fast",
        choices=["fast", "cv_bridge"],
        help="Decode ROS RGB8 images with np.frombuffer fast path or cv_bridge.",
    )
    parser.add_argument(
        "--max_wait_new_frame_ms",
        type=float,
        default=60.0,
        help="Before each request, wait up to this long if the latest RGB frame is stale. 0 disables.",
    )
    parser.add_argument(
        "--fresh_frame_age_threshold_ms",
        type=float,
        default=45.0,
        help="If latest RGB frame was received within this age, send immediately; otherwise wait for a fresher frame.",
    )
    parser.add_argument(
        "--latency_print_every",
        type=int,
        default=0,
        help="Print per-frame latency JSON every N plans. 0 disables per-frame latency logs; summary is still written.",
    )
    parser.add_argument("--enable_video_log", action="store_true", help="Record client-side overlay video.")
    parser.add_argument(
        "--save_demo_video",
        action="store_true",
        help="Save waypoint-overlay video to --demo_video_dir.",
    )
    parser.add_argument(
        "--demo_video_dir",
        type=str,
        default="output/demo_video",
        help="Directory for demo videos (waypoint_*.mp4).",
    )
    parser.add_argument(
        "--demo_fps",
        type=int,
        default=30,
        help="Demo video frame rate. Lower it (e.g. 15) if CPU contention with inference path is observed.",
    )
    parser.add_argument(
        "--record-video-pair",
        action="store_true",
        help="Record native RGB and the 384x384 waypoint/velocity overlay together.",
    )
    parser.add_argument(
        "--record-video-dir",
        type=str,
        default="output/videos",
        help="Output directory for native and model-overlay recordings.",
    )
    parser.add_argument(
        "--record-video-fps",
        type=int,
        default=15,
        help="Frame rate for both paired recordings.",
    )
    parser.add_argument(
        "--record-video-segment-seconds",
        type=float,
        default=10.0,
        help="Rotate paired MP4 files at this interval so a hard reset only loses the latest part.",
    )
    parser.add_argument(
        "--latency_log_path",
        type=str,
        default="",
        help="If set, write per-request latency samples + summary as JSON to this path on exit.",
    )
    parser.add_argument(
        "--latency_log_alias",
        type=str,
        default="",
        help="Optional second path for the same atomic latency log, such as output/live-latest.json.",
    )
    parser.add_argument(
        "--diagnostic-log-path",
        type=str,
        default="",
        help="Append crash-resilient inference and 10 Hz dispatch diagnostics as JSONL.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=os.environ.get("LOG_PATH", "output"),
        help="Directory to save client video logs",
    )
    parser.add_argument(
        "--waypoint-strategy",
        choices=WAYPOINT_STRATEGIES,
        default="first",
        help=(
            "Waypoint command selection: first=server baseline; two-step=direct wp1 for 0.1s "
            "then direct wp2 until replacement; dx4-dw1=average longitudinal commands from "
            "wp1..wp4 while taking yaw only from wp1."
        ),
    )
    parser.add_argument(
        "--vx-scale",
        type=float,
        default=None,
        help="Compatibility shortcut: set both positive and negative vx scales.",
    )
    parser.add_argument(
        "--vx-positive-scale",
        type=float,
        default=1.0,
        help="Multiply positive waypoint-derived vx before filtering and clipping.",
    )
    parser.add_argument(
        "--vx-negative-scale",
        type=float,
        default=1.0,
        help="Multiply negative waypoint-derived vx before filtering and clipping.",
    )
    parser.add_argument(
        "--wz-scale",
        type=float,
        default=1.0,
        help="Multiply waypoint-derived wz before filtering and final clipping. 1.0 preserves baseline.",
    )
    # ── Low-speed jitter mitigation (default = OFF, preserves baseline behavior) ──
    parser.add_argument(
        "--cmd-deadband-v",
        type=float,
        default=None,
        help="Compatibility shortcut: set both positive and negative vx deadbands.",
    )
    parser.add_argument(
        "--vx-positive-deadband",
        type=float,
        default=0.0,
        help="Zero positive vx below this threshold after scaling [m/s].",
    )
    parser.add_argument(
        "--vx-negative-deadband",
        type=float,
        default=0.0,
        help="Zero negative vx when its absolute value is below this threshold after scaling [m/s].",
    )
    parser.add_argument(
        "--cmd-deadband-w",
        type=float,
        default=0.0,
        help="Dead-band on commanded wz [rad/s]. |w|<dz -> send 0. 0=off. Typical 0.05-0.15.",
    )
    parser.add_argument(
        "--cmd-ema-alpha",
        type=float,
        default=1.0,
        help="EMA filter on dispatched (v, w). new = alpha*raw + (1-alpha)*prev. 1.0=off. 0.3=heavy smoothing.",
    )
    parser.add_argument(
        "--cmd-max-v",
        type=float,
        default=0.15,
        help="Final absolute vx limit before Unitree SDK dispatch [m/s].",
    )
    parser.add_argument(
        "--cmd-max-w",
        type=float,
        default=0.30,
        help="Final absolute yaw-rate limit before Unitree SDK dispatch [rad/s].",
    )
    parser.add_argument(
        "--cmd-hysteresis-v",
        type=float,
        default=0.0,
        help="Sign-flip hysteresis on vx: require |v|>this AND opposite sign to flip. 0=off. Typical 0.10-0.20.",
    )
    # ── Yaw boost: amplify model's turn intent when it's non-trivial.
    # Helps keep target inside camera FOV when person sidesteps quickly.
    # Only boosts |w| above threshold so it does NOT amplify low-speed noise.
    parser.add_argument(
        "--yaw-boost",
        type=float,
        default=1.0,
        help="Multiply commanded wz by this factor when |w| > yaw-boost-threshold. 1.0=off. Typical 1.3-1.8.",
    )
    parser.add_argument(
        "--yaw-boost-threshold",
        type=float,
        default=0.10,
        help="Boost only applies when |raw_w| > this [rad/s]. Below it the noise stays unamplified.",
    )
    parser.add_argument(
        "--yaw-boost-max",
        type=float,
        default=0.30,
        help="Final clamp on boosted wz [rad/s]. Should match the server max_wz so SDK doesn't over-clip.",
    )
    client_args, _ = parser.parse_known_args()

    bad_dry_run_args = [arg for arg in sys.argv[1:] if arg.startswith("--dry_run") and arg != "--dry_run"]
    if bad_dry_run_args:
        parser.error(f"invalid dry-run option(s): {bad_dry_run_args}; use exactly --dry_run")
    if client_args.dry_run and client_args.enable_live_control:
        parser.error("choose either --dry_run or --enable-live-control, not both")
    if not client_args.dry_run and not client_args.enable_live_control:
        parser.error("refusing live motion: use --dry_run, or explicitly pass --enable-live-control")
    if client_args.vx_scale is not None:
        client_args.vx_positive_scale = client_args.vx_scale
        client_args.vx_negative_scale = client_args.vx_scale
    if client_args.cmd_deadband_v is not None:
        client_args.vx_positive_deadband = client_args.cmd_deadband_v
        client_args.vx_negative_deadband = client_args.cmd_deadband_v
    for option, value in (
        ("--vx-positive-scale", client_args.vx_positive_scale),
        ("--vx-negative-scale", client_args.vx_negative_scale),
        ("--wz-scale", client_args.wz_scale),
        ("--vx-positive-deadband", client_args.vx_positive_deadband),
        ("--vx-negative-deadband", client_args.vx_negative_deadband),
    ):
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{option} must be a finite non-negative number")

    if client_args.camera_source == "go2" and client_args.send_depth:
        parser.error("--send_depth is only supported with --camera_source realsense")
    print(f"[conn] server_url={client_args.server_url}")

    http_session = requests.Session()
    http_session.trust_env = False
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
    http_session.mount("http://", adapter)
    start_diagnostic_writer(client_args.diagnostic_log_path)
    if client_args.transport in ("tcp_jpeg", "tcp_raw"):
        if not client_args.tcp_url:
            from urllib.parse import urlparse
            u = urlparse(client_args.server_url)
            client_args.tcp_url = f"tcp://{u.hostname or '127.0.0.1'}:5803"
        tcp_eval_client = TcpJpegEvalClient(client_args.tcp_url)

    if client_args.record_video_fps <= 0:
        parser.error("--record-video-fps must be positive")
    if client_args.record_video_segment_seconds <= 0.0:
        parser.error("--record-video-segment-seconds must be positive")

    LOG_PATH = client_args.log_path
    os.makedirs(LOG_PATH, exist_ok=True)
    if client_args.enable_video_log:
        video_path = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d_%H%M%S') + '.mp4')
        video_recorder = VideoRecorder(video_path, fps=24)
        video_recorder.start()
        init_frame = compose_waiting_frame(client_args.instruction)
        if init_frame is not None:
            video_recorder.update_frame(init_frame)
        print(f"[video] recording to {video_path}")

    if client_args.save_demo_video:
        os.makedirs(client_args.demo_video_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        wp_path = os.path.join(client_args.demo_video_dir, f'waypoint_{ts}.mp4')
        waypoint_recorder = VideoRecorder(wp_path, fps=int(client_args.demo_fps))
        waypoint_recorder.start()
        print(f"[demo] waypoint video -> {wp_path}")

    if client_args.record_video_pair:
        os.makedirs(client_args.record_video_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_name = str(client_args.camera_source).replace("/", "-")
        native_path = os.path.join(client_args.record_video_dir, f'{ts}_{source_name}_native.mp4')
        control_path = os.path.join(
            client_args.record_video_dir,
            f'{ts}_model384_waypoint_velocity.mp4',
        )
        native_video_recorder = VideoRecorder(
            native_path,
            fps=int(client_args.record_video_fps),
            segment_seconds=float(client_args.record_video_segment_seconds),
        )
        control_video_recorder = VideoRecorder(
            control_path,
            fps=int(client_args.record_video_fps),
            segment_seconds=float(client_args.record_video_segment_seconds),
        )
        native_video_recorder.start()
        control_video_recorder.start()
        print(
            f"[video-pair] native -> {native_path} (segmented); "
            f"model overlay -> {control_path} (segmented)"
        )

    # Set up shared memory for Unitree SDK communication
    odom_slots = 12 if client_args.diagnostic_log_path else 11
    odom_shm = mp.Array(ctypes.c_double, odom_slots, lock=False)
    cmd_shm = mp.Array(ctypes.c_double, 4, lock=False)
    unitree_proc = None
    odom_thread = None
    if not client_args.dry_run:
        # Start Unitree SDK in a child process (CycloneDDS, isolated from rclpy)
        control_interface = str(client_args.video_network_interface or "enP8p1s0")
        unitree_proc = mp.Process(
            target=unitree_process,
            args=(odom_shm, cmd_shm, control_interface),
            daemon=True,
        )
        unitree_proc.start()

        # Wait for first odom
        print("[main] Waiting for Unitree odom...")
        for _ in range(100):
            if not unitree_proc.is_alive():
                raise RuntimeError(
                    f"Unitree control DDS process exited on interface {control_interface}"
                )
            if odom_shm[0] > 0.5:
                print(f"[main] Got odom: x={odom_shm[1]:.3f} y={odom_shm[2]:.3f}")
                break
            time.sleep(0.1)
        else:
            cmd_shm[1] = 0.0
            cmd_shm[2] = 0.0
            cmd_shm[3] = 0.0
            cmd_shm[0] = -1.0
            time.sleep(0.3)
            unitree_proc.terminate()
            unitree_proc.join(timeout=2.0)
            raise RuntimeError(
                f"No Unitree SportModeState received on {control_interface} after 10s; "
                "refusing live control"
            )

        # Start odom polling thread
        odom_thread = threading.Thread(target=odom_poll_thread, daemon=True)
        odom_thread.start()

    planning_thread_instance = threading.Thread(target=planning_thread)
    dispatch_thread_instance = threading.Thread(target=cmd_dispatch_thread)
    planning_thread_instance.daemon = True
    dispatch_thread_instance.daemon = True
    video_render_thread_instance = None
    if client_args.enable_video_log:
        video_render_thread_instance = threading.Thread(target=video_render_thread)
        video_render_thread_instance.daemon = True
    demo_render_thread_instance = None
    if client_args.save_demo_video:
        demo_render_thread_instance = threading.Thread(target=demo_video_render_thread)
        demo_render_thread_instance.daemon = True
    paired_video_thread_instance = None
    if client_args.record_video_pair:
        paired_video_thread_instance = threading.Thread(target=paired_video_render_thread)
        paired_video_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager()

        planning_thread_instance.start()
        dispatch_thread_instance.start()
        if video_render_thread_instance is not None:
            video_render_thread_instance.start()
        if demo_render_thread_instance is not None:
            demo_render_thread_instance.start()
        if paired_video_thread_instance is not None:
            paired_video_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        # All worker loops observe this flag.  Set it before destroying the ROS
        # node or Unitree DDS objects so no daemon thread uses a torn-down client.
        exit_requested = True
        if manager is not None:
            manager.stop_video_client()
        for worker in (
            planning_thread_instance,
            dispatch_thread_instance,
            odom_thread,
            video_render_thread_instance,
            demo_render_thread_instance,
            paired_video_thread_instance,
        ):
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)
        try:
            print_latency_summary()
        except Exception as e:
            print(f"[latency] summary failed: {e}")
        if video_recorder is not None:
            video_recorder.stop()
        if waypoint_recorder is not None:
            waypoint_recorder.stop()
        if native_video_recorder is not None:
            native_video_recorder.stop()
        if control_video_recorder is not None:
            control_video_recorder.stop()
        if manager is not None:
            manager.destroy_node()
        if tcp_eval_client is not None:
            tcp_eval_client.close()
        if rclpy.ok():
            rclpy.shutdown()
        if unitree_proc is not None:
            cmd_shm[1] = 0.0
            cmd_shm[2] = 0.0
            cmd_shm[3] = 0.0
            cmd_shm[0] = -1.0
            time.sleep(0.4)
            unitree_proc.terminate()
            unitree_proc.join(timeout=2.0)
        stop_diagnostic_writer()
