from __future__ import annotations

from typing import List, MutableMapping

import torch
from transformers import PreTrainedModel

from ..config import ModelConfig
from ..modeling import MiniCPMRobotTrack
from .configuration import MiniCPMRobotTrackConfig


_LEGACY_PREFIX_MAP = (
    ("llm.", "backbone."),
    ("proj.net.", "vision_projector.layers."),
    ("tvi.time_emb.", "temporal_markers.time_embedding."),
    ("tvi.view_emb.", "temporal_markers.camera_embedding."),
    ("tvi.kind_emb.", "temporal_markers.stream_embedding."),
    ("planner.mlp.", "trajectory_head.layers."),
)
_LEGACY_UNUSED_PREFIXES = ("tvi.angle_proj.", "tvi.bbox_proj.")


def translate_legacy_hf_state_dict(
    state_dict: MutableMapping[str, torch.Tensor], prefix: str = ""
) -> None:
    """Translate the published legacy HF keys to the canonical package layout."""

    model_prefix = f"{prefix}model."
    if any(key.startswith(f"{model_prefix}backbone.") for key in state_dict):
        return

    translated = {}
    for key in list(state_dict):
        if not key.startswith(model_prefix):
            continue
        suffix = key[len(model_prefix) :]
        target = None
        if suffix == "act_token":
            target = "control_query"
        elif suffix == "alpha_task":
            target = "output_scale"
        elif suffix.startswith(_LEGACY_UNUSED_PREFIXES):
            state_dict.pop(key)
            continue
        else:
            for source_prefix, target_prefix in _LEGACY_PREFIX_MAP:
                if suffix.startswith(source_prefix):
                    target = target_prefix + suffix[len(source_prefix) :]
                    break
        if target is not None:
            translated[f"{model_prefix}{target}"] = state_dict.pop(key)
    state_dict.update(translated)


class MiniCPMRobotTrackForWaypoint(PreTrainedModel):
    """Hugging Face deployment wrapper around the canonical tracking model."""

    config_class = MiniCPMRobotTrackConfig
    base_model_prefix = "model"

    def __init__(self, config: MiniCPMRobotTrackConfig) -> None:
        super().__init__(config)
        model_config = ModelConfig(
            backbone_name=config.backbone_name,
            vision_feature_dim=config.vision_feature_dim,
            history_frames=config.history_frames,
            coarse_tokens_per_frame=config.coarse_tokens_per_frame,
            fine_tokens_current_frame=config.fine_tokens_current_frame,
            num_waypoints=config.num_waypoints,
            action_dim=config.action_dim,
            max_text_tokens=config.max_text_tokens,
            max_time_steps=config.max_time_steps,
            trajectory_dropout=config.trajectory_dropout,
            xy_scale=config.xy_scale,
            use_tanh_actions=config.use_tanh_actions,
            freeze_backbone=config.freeze_backbone,
            gradient_checkpointing=config.gradient_checkpointing,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = MiniCPMRobotTrack(model_config)
        self.post_init()

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
        instructions: List[str],
    ) -> torch.Tensor:
        return self.model(
            coarse_tokens,
            coarse_time_indices,
            fine_tokens,
            fine_time_indices,
            instructions,
        )

    @property
    def tokenizer(self):
        return self.model.tokenizer
