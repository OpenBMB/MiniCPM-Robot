#!/usr/bin/env python3
"""Waypoint -> (vx, vy, wz) velocity controller for on-device inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class WaypointVelocityControllerConfig:
    control_dt: float = 0.1
    max_vx: float = 0.6
    max_vy: float = 0.2
    max_wz: float = 0.6
    disable_lateral: bool = True
    controller_mode: str = "direct"
    direct_wp_idx: int = 1
    lookahead_idx: int = 2
    feedforward_points: int = 3
    ff_blend: float = 0.65
    kx: float = 1.2
    ky: float = 1.0
    ktheta: float = 1.4
    curvature_gain: float = 0.35
    min_turn_speed: float = 0.08
    allow_reverse: bool = False
    # ── Turn-in-place coupling (default OFF = 0.0): suppress vx when |wz| is large
    # so the dog can rotate without forward overshoot.  Useful when target moves
    # laterally faster than the dog can yaw at full vx.
    #
    #   vx_scale = max(min_vx_factor, 1 - turn_in_place_factor * |wz|/max_wz)
    #
    #   turn_in_place_factor=0:   off (production default)
    #   turn_in_place_factor=1:   at full yaw, vx drops to `min_vx_factor` of nominal
    turn_in_place_factor: float = 0.0
    min_vx_factor: float = 0.2


class WaypointVelocityController:
    """Map predicted local-frame waypoints to robot velocity commands."""

    def __init__(self, cfg: WaypointVelocityControllerConfig):
        self.cfg = cfg

    def _clip_cmd(self, vx: float, vy: float, wz: float) -> np.ndarray:
        if self.cfg.disable_lateral:
            vy = 0.0
        # Clamp wz first so the coupling uses the actual yaw rate the dog will see.
        wz_clipped = float(np.clip(wz, -self.cfg.max_wz, self.cfg.max_wz))
        # Optional: suppress vx when |wz| is large (turn-in-place coupling).
        if self.cfg.turn_in_place_factor > 0.0 and self.cfg.max_wz > 0.0:
            wz_norm = abs(wz_clipped) / self.cfg.max_wz
            scale = 1.0 - self.cfg.turn_in_place_factor * wz_norm
            scale = max(self.cfg.min_vx_factor, scale)
            vx *= scale
        return np.array(
            [
                float(np.clip(vx, -self.cfg.max_vx, self.cfg.max_vx)),
                float(np.clip(vy, -self.cfg.max_vy, self.cfg.max_vy)),
                wz_clipped,
            ],
            dtype=np.float32,
        )

    def _direct(self, waypoints: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        dt = max(1e-6, float(self.cfg.control_dt))
        wp_idx = min(max(1, int(self.cfg.direct_wp_idx)), int(waypoints.shape[0]) - 1)
        wp = waypoints[wp_idx]
        vx = float(wp[0]) / (wp_idx * dt)
        vy = float(wp[1]) / (wp_idx * dt) if wp.shape[0] >= 2 else 0.0
        wz = float(wp[2]) / (wp_idx * dt) if wp.shape[0] >= 3 else 0.0
        return self._clip_cmd(vx, vy, wz), {
            "mode": "direct",
            "used_wp_idx": int(wp_idx),
        }

    def _lookahead(self, waypoints: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        dt = max(1e-6, float(self.cfg.control_dt))
        n_wp = int(waypoints.shape[0])
        lookahead_idx = min(max(1, int(self.cfg.lookahead_idx)), n_wp - 1)
        ff_points = min(max(1, int(self.cfg.feedforward_points)), n_wp - 1)

        ff_vels = []
        ff_weights = []
        for idx in range(1, ff_points + 1):
            wp = waypoints[idx]
            ff_vels.append(
                [
                    float(wp[0]) / (idx * dt),
                    float(wp[1]) / (idx * dt) if wp.shape[0] >= 2 else 0.0,
                    float(wp[2]) / (idx * dt) if wp.shape[0] >= 3 else 0.0,
                ]
            )
            ff_weights.append(1.0 / idx)
        ff_vel = np.average(np.asarray(ff_vels, dtype=np.float32), axis=0, weights=np.asarray(ff_weights))

        target = waypoints[lookahead_idx]
        tx = float(target[0])
        ty = float(target[1]) if target.shape[0] >= 2 else 0.0
        ttheta = float(target[2]) if target.shape[0] >= 3 else 0.0
        dist = math.hypot(tx, ty)

        vx_fb = self.cfg.kx * tx
        vy_fb = self.cfg.ky * ty if not self.cfg.disable_lateral else 0.0
        curvature = 0.0 if dist < 1e-6 else (2.0 * ty) / max(1e-6, dist * dist)
        base_vx = max(float(ff_vel[0]), self.cfg.min_turn_speed)
        wz_fb = self.cfg.ktheta * _wrap_to_pi(ttheta) + self.cfg.curvature_gain * base_vx * curvature

        blend = float(np.clip(self.cfg.ff_blend, 0.0, 1.0))
        vx = blend * float(ff_vel[0]) + (1.0 - blend) * vx_fb
        vy = blend * float(ff_vel[1]) + (1.0 - blend) * vy_fb
        wz = blend * float(ff_vel[2]) + (1.0 - blend) * wz_fb

        if not self.cfg.allow_reverse:
            vx = max(0.0, vx)

        return self._clip_cmd(vx, vy, wz), {
            "mode": "lookahead",
            "used_wp_idx": int(lookahead_idx),
            "feedforward_points": int(ff_points),
            "target_x": tx,
            "target_y": ty,
            "target_theta": ttheta,
            "target_dist": dist,
            "curvature": curvature,
            "ff_velocity": [float(v) for v in ff_vel.tolist()],
        }

    def compute(self, waypoints: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        if waypoints.ndim != 2 or waypoints.shape[0] < 2:
            return np.zeros(3, dtype=np.float32), {"mode": "invalid", "reason": "need at least 2 waypoints"}
        mode = str(self.cfg.controller_mode).strip().lower()
        if mode == "lookahead":
            return self._lookahead(waypoints)
        return self._direct(waypoints)
