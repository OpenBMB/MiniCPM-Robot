from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..checkpoint import load_model
from ..vision import DualVisionEncoder, VisionEncoderConfig


class MiniCPMRobotTrackPolicy:
    """Stateful RGB policy used by the Habitat evaluation loop."""

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        backbone_override: Optional[str] = None,
        control_dt: float = 0.1,
        control_waypoint: int = 1,
    ) -> None:
        self.device = device
        self._uses_hf_snapshot = checkpoint.is_dir()
        self.tokenizer = None
        if self._uses_hf_snapshot:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModel.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                local_files_only=True,
            ).eval().to(device)
        else:
            self.model, _ = load_model(
                checkpoint, device=device, backbone_override=backbone_override
            )
        self.encoder = DualVisionEncoder(VisionEncoderConfig(), device=device)
        self.control_dt = control_dt
        self.control_waypoint = control_waypoint
        if not 0 <= control_waypoint < self.model.config.num_waypoints:
            raise ValueError("control waypoint index is outside the predicted trajectory")
        self._history: Deque[torch.Tensor] = deque(
            maxlen=self.model.config.history_frames
        )

    def reset(self) -> None:
        self._history.clear()

    @torch.inference_mode()
    def act(self, rgb: np.ndarray, instruction: Optional[str]) -> List[float]:
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError(f"expected an RGB image, got shape {rgb.shape}")
        image = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
        coarse, fine = self.encoder.encode_frame(
            image,
            coarse_tokens=self.model.config.coarse_tokens_per_frame,
            fine_tokens=self.model.config.fine_tokens_current_frame,
        )
        if coarse.size(-1) != self.model.config.vision_feature_dim:
            raise ValueError(
                "visual feature size does not match the training checkpoint"
            )

        history_frames = self.model.config.history_frames
        if history_frames > 0:
            self._history.append(coarse.cpu())
            values = list(self._history)
            values = [values[0]] * (history_frames - len(values)) + values
            coarse_batch = torch.cat(values, dim=0).unsqueeze(0).to(self.device)
            coarse_times = torch.arange(history_frames, device=self.device)
            coarse_times = coarse_times.repeat_interleave(
                self.model.config.coarse_tokens_per_frame
            ).unsqueeze(0)
        else:
            coarse_batch = coarse.new_empty((1, 0, coarse.size(-1))).to(self.device)
            coarse_times = torch.empty((1, 0), dtype=torch.long, device=self.device)

        fine_batch = fine.unsqueeze(0).to(self.device)
        fine_times = torch.full(
            (1, fine.size(0)), history_frames, dtype=torch.long, device=self.device
        )
        prompt = [instruction or "Follow the target person."]
        if self._uses_hf_snapshot:
            if self.tokenizer is None:
                raise RuntimeError("model tokenizer is not initialized")
            text = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.model.config.max_text_tokens),
            )
            outputs = self.model(
                input_ids=text["input_ids"].to(self.device),
                attention_mask=text["attention_mask"].to(self.device),
                coarse_tokens=coarse_batch,
                coarse_time_indices=coarse_times,
                fine_tokens=fine_batch,
                fine_time_indices=fine_times,
            )
            trajectory = outputs.trajectories
        else:
            trajectory = self.model(
                coarse_batch,
                coarse_times,
                fine_batch,
                fine_times,
                prompt,
            )
        waypoint = trajectory[0, self.control_waypoint].float().cpu()
        # Match the released benchmark policy: convert the float32 waypoint to
        # a Python float before applying dt, rather than dividing in float32.
        return [float(value.item()) / self.control_dt for value in waypoint]
