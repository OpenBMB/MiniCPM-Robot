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

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .config import ModelConfig
from .modeling import MiniCPMRobotTrack


_LEGACY_PREFIX_MAP = (
    ("llm.", "backbone."),
    ("proj.net.", "vision_projector.layers."),
    ("tvi.time_emb.", "temporal_markers.time_embedding."),
    ("tvi.view_emb.", "temporal_markers.camera_embedding."),
    ("tvi.kind_emb.", "temporal_markers.stream_embedding."),
    ("planner.mlp.", "trajectory_head.layers."),
)

_LEGACY_UNUSED_PREFIXES = ("tvi.angle_proj.", "tvi.bbox_proj.")


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    return value


def save_checkpoint(
    path: Path,
    model: MiniCPMRobotTrack,
    *,
    epoch: int,
    global_step: int,
    training_config: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "model_name": "MiniCPM-RobotTrack",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "training_config": training_config or {},
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def _legacy_config(
    values: Dict[str, Any], backbone_override: Optional[str]
) -> ModelConfig:
    planner_type = str(values.get("planner_type", "funnel")).lower()
    if planner_type != "funnel":
        raise ValueError(
            f"legacy checkpoint uses unsupported planner_type={planner_type!r}; "
            "this release supports the published funnel checkpoint"
        )

    configured_backbone = str(values.get("llm_name", "")).strip()
    if backbone_override is not None:
        backbone_name = backbone_override
    elif configured_backbone and not Path(configured_backbone).is_absolute():
        backbone_name = configured_backbone
    else:
        # Training checkpoints can contain a machine-local Hugging Face snapshot.
        backbone_name = "openbmb/MiniCPM4-0.5B"

    alpha_xy = values.get("alpha_xy", 2.0)
    return ModelConfig(
        backbone_name=backbone_name,
        vision_feature_dim=int(values.get("vision_feat_dim", 1536)),
        history_frames=int(values.get("history", 31)),
        coarse_tokens_per_frame=int(values.get("coarse_tokens", 4)),
        fine_tokens_current_frame=int(values.get("fine_tokens", 64)),
        num_waypoints=int(values.get("n_waypoints", 8)),
        max_time_steps=int(values.get("max_time", 4096)),
        trajectory_dropout=float(values.get("planner_dropout", 0.4)),
        xy_scale=float(2.0 if alpha_xy is None else alpha_xy),
        # Match the released TrackVLA benchmark entry: legacy checkpoints were
        # evaluated with the funnel head's default tanh enabled, regardless of
        # the training-only no_tanh_actions flag stored in the checkpoint.
        use_tanh_actions=True,
    )


def _translate_legacy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    translated: Dict[str, Any] = {}
    ignored = []
    for original_key, value in state.items():
        key = original_key.removeprefix("module.")
        if key == "act_token":
            translated["control_query"] = value
            continue
        if key == "alpha_task":
            translated["output_scale"] = value
            continue
        if key.startswith(_LEGACY_UNUSED_PREFIXES):
            ignored.append(key)
            continue
        for source_prefix, target_prefix in _LEGACY_PREFIX_MAP:
            if key.startswith(source_prefix):
                translated[target_prefix + key[len(source_prefix) :]] = value
                break
        else:
            raise ValueError(f"unsupported legacy checkpoint key: {original_key}")

    if len(ignored) != 4:
        raise ValueError(
            "legacy checkpoint must contain the four unused angle/bbox TVI tensors"
        )
    return translated


def _load_legacy_model(
    payload: Dict[str, Any],
    *,
    device: torch.device,
    backbone_override: Optional[str],
) -> Tuple[MiniCPMRobotTrack, Dict[str, Any]]:
    config_values = payload.get("config")
    state = payload.get("model_state") or payload.get("model_state_dict")
    if not isinstance(config_values, dict) or not isinstance(state, dict):
        raise ValueError("legacy checkpoint is missing config or model_state")

    config = _legacy_config(config_values, backbone_override)
    model = MiniCPMRobotTrack(config)
    translated = _translate_legacy_state(state)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(translated))
    unexpected = sorted(set(translated) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(translated)
        if tuple(expected[key].shape) != tuple(translated[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "legacy checkpoint does not match the published funnel model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={mismatched[:8]}"
        )
    model.load_state_dict(translated, strict=True)
    model.to(device)
    model.eval()
    payload = dict(payload)
    payload["model_name"] = "MiniCPM-RobotTrack"
    payload["model_config"] = config.to_dict()
    payload["format_version"] = 0
    return model, payload


def load_model(
    path: Path,
    *,
    device: torch.device,
    backbone_override: Optional[str] = None,
) -> Tuple[MiniCPMRobotTrack, Dict[str, Any]]:
    payload = _torch_load(path)
    if payload.get("model_name") != "MiniCPM-RobotTrack":
        if "model_state" in payload and "config" in payload:
            return _load_legacy_model(
                payload, device=device, backbone_override=backbone_override
            )
        raise ValueError(f"unsupported checkpoint model name in {path}")
    config_values = payload.get("model_config")
    state = payload.get("model_state")
    if not isinstance(config_values, dict) or not isinstance(state, dict):
        raise ValueError(f"checkpoint is missing model_config or model_state: {path}")

    config = ModelConfig.from_dict(config_values)
    if backbone_override is not None:
        config.backbone_name = backbone_override
        config.validate()
    model = MiniCPMRobotTrack(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, payload
