from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


def _square_side(token_count: int) -> int:
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"token count {token_count} does not form a square grid")
    return side


def grid_pool_tokens(
    patch_tokens: torch.Tensor,
    grid_height: int,
    grid_width: int,
    output_tokens: int,
) -> torch.Tensor:
    """Average-pools patch tokens to a fixed square token grid."""

    batch, patch_count, channels = patch_tokens.shape
    if patch_count != grid_height * grid_width:
        raise ValueError("patch token count does not match the supplied grid")
    output_side = _square_side(output_tokens)
    features = patch_tokens.transpose(1, 2).reshape(
        batch, channels, grid_height, grid_width
    )
    features = F.adaptive_avg_pool2d(features, (output_side, output_side))
    return features.flatten(2).transpose(1, 2).contiguous()


def _resize_token_grid(tokens: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    batch, token_count, channels = tokens.shape
    side = _square_side(token_count)
    features = tokens.transpose(1, 2).reshape(batch, channels, side, side)
    features = F.adaptive_avg_pool2d(features, output_size)
    return features.flatten(2).transpose(1, 2).contiguous()


@dataclass(frozen=True)
class VisionEncoderConfig:
    dino_model_name: str = field(
        default_factory=lambda: os.environ.get(
            "DINOV3_MODEL_PATH", "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
    )
    siglip_model_name: str = field(
        default_factory=lambda: os.environ.get(
            "SIGLIP_MODEL_PATH", "google/siglip-so400m-patch14-384"
        )
    )
    image_size: int = 384


class DualVisionEncoder(nn.Module):
    """Frozen DINOv3 + SigLIP encoder used by training and evaluation."""

    def __init__(self, config: VisionEncoderConfig, device: torch.device) -> None:
        super().__init__()
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            SiglipImageProcessor,
            SiglipVisionModel,
        )

        self.config = config
        self.device = device
        self.dino_processor = AutoImageProcessor.from_pretrained(config.dino_model_name)
        self.dino = AutoModel.from_pretrained(config.dino_model_name).eval().to(device)
        self.dino_register_tokens = int(
            getattr(self.dino.config, "num_register_tokens", 0) or 0
        )
        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            config.siglip_model_name
        )
        self.siglip = SiglipVisionModel.from_pretrained(config.siglip_model_name).eval().to(device)
        self.requires_grad_(False)

    def _prepare(self, images: Sequence[Image.Image], processor) -> dict:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        resized = [
            image.convert("RGB").resize(
                (self.config.image_size, self.config.image_size), resampling
            )
            for image in images
        ]
        values = processor(
            images=resized,
            return_tensors="pt",
            size={"height": self.config.image_size, "width": self.config.image_size},
        )
        return {key: value.to(self.device) for key, value in values.items()}

    @torch.inference_mode()
    def encode_full(self, images: Sequence[Image.Image]) -> Tuple[torch.Tensor, int, int]:
        if not images:
            raise ValueError("at least one image is required")

        dino_output = self.dino(**self._prepare(images, self.dino_processor))
        dino_tokens = dino_output.last_hidden_state[
            :, 1 + self.dino_register_tokens :, :
        ]
        grid_side = _square_side(dino_tokens.size(1))

        siglip_output = self.siglip(**self._prepare(images, self.siglip_processor))
        siglip_tokens = siglip_output.last_hidden_state
        if _square_side_or_none(siglip_tokens.size(1)) is None:
            if _square_side_or_none(siglip_tokens.size(1) - 1) is None:
                raise ValueError("SigLIP output does not contain a square patch grid")
            siglip_tokens = siglip_tokens[:, 1:, :]
        siglip_tokens = _resize_token_grid(siglip_tokens, (grid_side, grid_side))

        combined = torch.cat((dino_tokens, siglip_tokens), dim=-1)
        return combined, grid_side, grid_side

    @torch.inference_mode()
    def encode_pooled(
        self,
        images: Sequence[Image.Image],
        *,
        coarse_tokens: int = 4,
        fine_tokens: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full, grid_height, grid_width = self.encode_full(images)
        coarse = grid_pool_tokens(full, grid_height, grid_width, coarse_tokens)
        fine = grid_pool_tokens(full, grid_height, grid_width, fine_tokens)
        return coarse.float(), fine.float()

    @torch.inference_mode()
    def encode_frame(
        self,
        image: Image.Image,
        *,
        coarse_tokens: int = 4,
        fine_tokens: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse, fine = self.encode_pooled(
            [image], coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
        )
        return coarse[0], fine[0]


def _square_side_or_none(token_count: int):
    if token_count <= 0:
        return None
    side = int(round(math.sqrt(token_count)))
    return side if side * side == token_count else None
