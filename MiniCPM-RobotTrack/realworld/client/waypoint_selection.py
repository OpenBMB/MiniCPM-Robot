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

"""Client-side waypoint command strategies retained after live evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


WAYPOINT_STRATEGIES = (
    "first",
    "two-step",
    "dx4-dw1",
)


def scale_velocity_command(
    vx: float,
    wz: float,
    vx_positive_scale: float = 1.0,
    vx_negative_scale: float = 1.0,
    wz_scale: float = 1.0,
) -> Tuple[float, float]:
    """Apply direction-aware vx gain before filtering and safety clipping."""
    vx = float(vx)
    vx_scale = vx_positive_scale if vx >= 0.0 else vx_negative_scale
    return vx * float(vx_scale), float(wz) * float(wz_scale)


def apply_directional_vx_deadband(
    vx: float,
    positive_deadband: float = 0.0,
    negative_deadband: float = 0.0,
) -> float:
    """Zero vx using a threshold selected from the command direction."""
    vx = float(vx)
    threshold = positive_deadband if vx >= 0.0 else negative_deadband
    return 0.0 if abs(vx) < float(threshold) else vx


def _normalize_waypoints(
    waypoints: Iterable[Sequence[float]],
) -> List[Tuple[float, float, float]]:
    normalized = []
    for waypoint in waypoints:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 3:
            raise ValueError("each waypoint must contain x, y and yaw")
        item = (float(waypoint[0]), float(waypoint[1]), float(waypoint[2]))
        if not all(math.isfinite(value) for value in item):
            raise ValueError("waypoints must contain finite values")
        normalized.append(item)
    if len(normalized) < 2:
        raise ValueError("at least two waypoints are required")
    return normalized


def direct_waypoint_velocity(
    waypoint: Sequence[float],
    index: int,
    control_dt: float,
) -> Tuple[float, float]:
    """Match the server direct controller for one cumulative waypoint."""
    index = max(1, int(index))
    dt = max(1e-6, float(control_dt))
    horizon = index * dt
    return float(waypoint[0]) / horizon, float(waypoint[2]) / horizon


def two_step_velocity_plan(
    waypoints: Iterable[Sequence[float]],
    control_dt: float,
) -> List[Tuple[float, float]]:
    """Return direct-controller commands for cumulative waypoint 1 then 2."""
    points = _normalize_waypoints(waypoints)
    if len(points) < 3:
        raise ValueError("two-step strategy requires at least three waypoints")
    return [direct_waypoint_velocity(points[index], index, control_dt) for index in (1, 2)]


def two_step_velocity_at_elapsed(
    plan: Sequence[Sequence[float]],
    elapsed: float,
    control_dt: float,
) -> Tuple[float, float, int]:
    """Use wp1 for the first interval, then hold wp2 until plan replacement."""
    if len(plan) < 2 or len(plan[0]) < 2 or len(plan[1]) < 2:
        raise ValueError("two-step plan must contain two vx/wz commands")
    index = 0 if max(0.0, float(elapsed)) < max(1e-6, float(control_dt)) else 1
    return float(plan[index][0]), float(plan[index][1]), index


def dx4_dw1_velocity(
    waypoints: Iterable[Sequence[float]],
    control_dt: float,
):
    """Average longitudinal velocity from wp1..wp4; take yaw only from wp1."""
    points = _normalize_waypoints(waypoints)
    if len(points) < 5:
        raise ValueError("dx4-dw1 strategy requires at least five waypoints")
    longitudinal_candidates = [
        direct_waypoint_velocity(points[index], index, control_dt)[0]
        for index in range(1, 5)
    ]
    _, wz = direct_waypoint_velocity(points[1], 1, control_dt)
    vx = sum(longitudinal_candidates) / len(longitudinal_candidates)
    return float(vx), float(wz), {
        "waypoint_selected_index": 4,
        "waypoint_selected_segment": -1,
        "waypoint_blend_alpha": 0.0,
        "waypoint_fused_indices": [1, 2, 3, 4],
        "waypoint_fused_vx": [float(value) for value in longitudinal_candidates],
        "waypoint_fused_vx_mean": float(vx),
        "waypoint_yaw_index": 1,
    }


def _base_info(strategy: str, points, age_s: float, dt: float) -> Dict[str, Any]:
    return {
        "waypoint_strategy": str(strategy),
        "waypoint_count": int(len(points)),
        "waypoint_age_ms": max(0.0, float(age_s)) * 1000.0,
        "waypoint_latency_steps": max(0.0, float(age_s)) / dt,
        "waypoint_selected_index": -1,
        "waypoint_selected_segment": -1,
        "waypoint_blend_alpha": 0.0,
    }


def select_waypoint_velocity(
    waypoints: Iterable[Sequence[float]],
    strategy: str,
    age_s: float,
    control_dt: float,
):
    """Select a single command for first or dx4-dw1."""
    points = _normalize_waypoints(waypoints)
    dt = max(1e-6, float(control_dt))
    strategy = str(strategy).strip().lower()
    if strategy not in WAYPOINT_STRATEGIES:
        raise ValueError(f"unknown waypoint strategy: {strategy}")
    info = _base_info(strategy, points, age_s, dt)

    if strategy in ("first", "two-step"):
        vx, wz = direct_waypoint_velocity(points[1], 1, dt)
        info.update({
            "waypoint_selected_index": 1,
            "waypoint_selected_x": float(points[1][0]),
            "waypoint_selected_y": float(points[1][1]),
            "waypoint_selected_yaw": float(points[1][2]),
        })
        return vx, wz, info

    vx, wz, fusion_info = dx4_dw1_velocity(points, dt)
    info.update(fusion_info)
    return vx, wz, info
