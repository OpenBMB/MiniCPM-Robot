from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import ModelConfig


class VisionProjector(nn.Module):
    """Maps concatenated DINOv3 and SigLIP features into MiniCPM space."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TemporalMarkerEncoder(nn.Module):
    """Builds one marker token for each frame represented in the sequence."""

    def __init__(self, hidden_dim: int, max_time_steps: int) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(max_time_steps, hidden_dim)
        self.stream_embedding = nn.Embedding(2, hidden_dim)
        self.camera_embedding = nn.Embedding(1, hidden_dim)

    def forward(self, time_step: int, stream_id: int, device: torch.device) -> torch.Tensor:
        time = torch.tensor([time_step], dtype=torch.long, device=device)
        stream = torch.tensor([stream_id], dtype=torch.long, device=device)
        camera = torch.zeros(1, dtype=torch.long, device=device)
        return (
            self.time_embedding(time)
            + self.stream_embedding(stream)
            + self.camera_embedding(camera)
        ).squeeze(0)


class FunnelTrajectoryHead(nn.Module):
    """Six-layer funnel MLP that predicts a fixed waypoint trajectory."""

    def __init__(
        self,
        hidden_dim: int,
        num_waypoints: int,
        action_dim: int,
        dropout: float,
        use_tanh: bool,
    ) -> None:
        super().__init__()
        output_dim = num_waypoints * action_dim
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, control_state: torch.Tensor) -> torch.Tensor:
        trajectory = self.layers(control_state)
        if self.use_tanh:
            trajectory = torch.tanh(trajectory)
        return trajectory.view(-1, self.num_waypoints, self.action_dim)


def _backbone_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError("MiniCPM configuration does not expose hidden_size")
    return int(hidden_size)


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


class MiniCPMRobotTrack(nn.Module):
    """MiniCPM visual tracking policy with a funnel trajectory head."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        backbone: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        if backbone is None or tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            load_dtype = torch.bfloat16 if torch.cuda.is_available() else None
            if backbone is None:
                backbone = AutoModel.from_pretrained(
                    config.backbone_name,
                    torch_dtype=load_dtype,
                    trust_remote_code=config.trust_remote_code,
                )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.backbone_name,
                    trust_remote_code=config.trust_remote_code,
                )

        self.backbone = backbone
        self.tokenizer = tokenizer
        hidden_dim = _backbone_hidden_size(self.backbone)

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token

        self._configure_backbone()
        self.vision_projector = VisionProjector(config.vision_feature_dim, hidden_dim)
        self.temporal_markers = TemporalMarkerEncoder(hidden_dim, config.max_time_steps)
        self.control_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.control_query, mean=0.0, std=0.02)
        self.trajectory_head = FunnelTrajectoryHead(
            hidden_dim=hidden_dim,
            num_waypoints=config.num_waypoints,
            action_dim=config.action_dim,
            dropout=config.trajectory_dropout,
            use_tanh=config.use_tanh_actions,
        )

        output_scale = torch.ones(1, 1, config.action_dim, dtype=torch.float32)
        output_scale[..., :2] = config.xy_scale
        self.register_buffer("output_scale", output_scale)

    def _configure_backbone(self) -> None:
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False

        if self.config.gradient_checkpointing:
            enable = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if enable is not None:
                try:
                    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enable()
            require_input_grads = getattr(self.backbone, "enable_input_require_grads", None)
            if require_input_grads is not None:
                require_input_grads()

        self.backbone.requires_grad_(not self.config.freeze_backbone)

    def _embed_text(self, instructions: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(instructions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_text_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        return embeddings, attention_mask

    def _insert_temporal_markers(
        self,
        tokens: torch.Tensor,
        time_indices: torch.Tensor,
        stream_id: int,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or time_indices.ndim != 2:
            raise ValueError("visual tokens and time indices must have shapes [B, N, C] and [B, N]")
        if tokens.shape[:2] != time_indices.shape:
            raise ValueError("visual token and time-index shapes do not match")
        if tokens.size(1) == 0:
            return tokens

        packed_rows: List[torch.Tensor] = []
        time_rows = time_indices.detach().to("cpu").tolist()
        for batch_index, time_row in enumerate(time_rows):
            pieces: List[torch.Tensor] = []
            start = 0
            while start < len(time_row):
                time_step = int(time_row[start])
                if not 0 <= time_step < self.config.max_time_steps:
                    raise ValueError(f"time index {time_step} is outside the configured range")
                end = start + 1
                while end < len(time_row) and int(time_row[end]) == time_step:
                    end += 1
                marker = self.temporal_markers(time_step, stream_id, tokens.device)
                pieces.extend((marker.unsqueeze(0), tokens[batch_index, start:end]))
                start = end
            packed_rows.append(torch.cat(pieces, dim=0))

        packed_lengths = {row.size(0) for row in packed_rows}
        if len(packed_lengths) != 1:
            raise ValueError("each batch item must contain the same number of represented frames")
        return torch.stack(packed_rows, dim=0)

    def _build_sequence(
        self,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
        instructions: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.control_query.device
        batch_size = coarse_tokens.size(0)
        if fine_tokens.size(0) != batch_size or len(instructions) != batch_size:
            raise ValueError("batch dimensions do not match")

        projector_dtype = _module_dtype(self.vision_projector)
        history = self.vision_projector(coarse_tokens.to(device=device, dtype=projector_dtype))
        current = self.vision_projector(fine_tokens.to(device=device, dtype=projector_dtype))
        history = self._insert_temporal_markers(history, coarse_time_indices.to(device), stream_id=0)
        current = self._insert_temporal_markers(current, fine_time_indices.to(device), stream_id=1)
        text, text_mask = self._embed_text(instructions, device)

        control_query = self.control_query.expand(batch_size, -1, -1)
        sequence = torch.cat((text, history, current, control_query), dim=1)
        backbone_dtype = _module_dtype(self.backbone)
        sequence = sequence.to(dtype=backbone_dtype)

        attention_mask = torch.cat(
            (
                text_mask,
                torch.ones(batch_size, history.size(1), dtype=torch.long, device=device),
                torch.ones(batch_size, current.size(1), dtype=torch.long, device=device),
                torch.ones(batch_size, 1, dtype=torch.long, device=device),
            ),
            dim=1,
        )
        return sequence, attention_mask

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        sequence, attention_mask = self._build_sequence(
            coarse_tokens,
            coarse_time_indices,
            fine_tokens,
            fine_time_indices,
            instructions,
        )
        output = self.backbone(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = getattr(output, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = output[0]
        control_state = hidden_states[:, -1].to(dtype=_module_dtype(self.trajectory_head))
        normalized_trajectory = self.trajectory_head(control_state)
        return normalized_trajectory * self.output_scale.to(normalized_trajectory.dtype)

    def normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory / self.output_scale.to(device=trajectory.device, dtype=trajectory.dtype)
