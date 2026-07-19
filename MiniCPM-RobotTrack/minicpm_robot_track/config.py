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

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Architecture settings stored with every training checkpoint."""

    backbone_name: str = "openbmb/MiniCPM4-0.5B"
    vision_feature_dim: int = 1536
    history_frames: int = 31
    coarse_tokens_per_frame: int = 4
    fine_tokens_current_frame: int = 64
    num_waypoints: int = 8
    action_dim: int = 3
    max_text_tokens: int = 128
    max_time_steps: int = 4096
    trajectory_dropout: float = 0.4
    xy_scale: float = 2.0
    use_tanh_actions: bool = True
    freeze_backbone: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = True

    def validate(self) -> None:
        if not self.backbone_name.strip():
            raise ValueError("backbone_name cannot be empty")
        if self.vision_feature_dim <= 0:
            raise ValueError("vision_feature_dim must be positive")
        if self.history_frames < 0:
            raise ValueError("history_frames cannot be negative")
        for name, value in (
            ("coarse_tokens_per_frame", self.coarse_tokens_per_frame),
            ("fine_tokens_current_frame", self.fine_tokens_current_frame),
        ):
            side = int(round(value**0.5)) if value > 0 else 0
            if side * side != value:
                raise ValueError(f"{name} must be a positive square number")
        if self.num_waypoints < 2:
            raise ValueError("num_waypoints must be at least 2")
        if self.action_dim != 3:
            raise ValueError("MiniCPM-RobotTrack expects actions in [x, y, yaw] format")
        if self.max_text_tokens <= 0 or self.max_time_steps <= 0:
            raise ValueError("token and time limits must be positive")
        if not 0.0 <= self.trajectory_dropout < 1.0:
            raise ValueError("trajectory_dropout must be in [0, 1)")
        if self.xy_scale <= 0.0:
            raise ValueError("xy_scale must be positive")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        known = {item.name for item in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in known})
        config.validate()
        return config
