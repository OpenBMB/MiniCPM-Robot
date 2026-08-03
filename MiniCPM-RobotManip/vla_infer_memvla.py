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

"""memVLA inference: visual history in, sub-task and one action chunk out.

The RMBench recipe runs the checkpoint in two stages:

1. Greedily generate the current sub-task as the assistant answer, conditioned on
   the recent history video from three cameras.
2. Append the generated tokens to the prompt and run the action head on the
   hidden states of that full sequence.

Stage 2 is exactly what ``MiniCPMV_VLA.predict_action`` already does -- it takes
``hidden_states[-1]`` of the whole sequence -- which is the ``condition_span="full"``
training recipe. No checkpoint or remote-code change is required; this module only
drives the released model's public surface.

Prompt construction is frozen in ``deployment/model_server/memvla_recipe.py`` and
must stay byte-identical to training: the action head is conditioned on the hidden
states of this sequence, so prompt drift silently degrades closed-loop behaviour.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from deployment.model_server import memvla_recipe as recipe


@contextlib.contextmanager
def _cache_vision_encoding(vlm):
    """Memoize the vision encoder for the duration of one generation.

    Sub-task decoding re-runs the full forward pass per token, and each pass
    re-encodes all 62 history frames through the ViT -- about 62% of a single
    predict, repeated once per generated token. Within one predict the
    ``pixel_values`` tensor is identical and the encoder is deterministic, so
    memoizing on the underlying buffer is a mathematical identity: the vision
    tower runs once and every other code path is untouched.
    """
    original = vlm.get_image_features
    cache: dict = {}

    def cached(pixel_values, target_sizes, downsample_mode=None):
        key = (pixel_values.data_ptr(), tuple(pixel_values.shape), int(target_sizes.sum()))
        if key not in cache:
            cache[key] = original(pixel_values, target_sizes, downsample_mode=downsample_mode)
        return cache[key]

    vlm.get_image_features = cached
    try:
        yield
    finally:
        vlm.get_image_features = original


class MemVLAInference:
    """Processor and model wrapper for single-sample memVLA inference."""

    def __init__(
        self,
        checkpoint_path: str | Path = "./checkpoint",
        device: str | torch.device | None = None,
        max_subtask_tokens: int = recipe.MAX_SUBTASK_TOKENS,
        cache_vision: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = str(checkpoint_path)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.model.to(self.device).eval()
        self.max_subtask_tokens = int(max_subtask_tokens)
        self.cache_vision = bool(cache_vision)

        tokenizer = self.processor.tokenizer
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.im_end_id, int) or self.im_end_id < 0:
            raise ValueError("tokenizer has no <|im_end|>; sub-task generation cannot terminate")

    @staticmethod
    def _load_views(views: Sequence[Sequence]) -> list[list[np.ndarray]]:
        """Validate the per-view frame counts and resize every frame to 448x448."""
        recipe.validate_views([len(view) for view in views])
        loaded = []
        for view in views:
            frames = []
            for frame in view:
                if isinstance(frame, (str, Path)):
                    with Image.open(frame) as pil_image:
                        array = np.asarray(pil_image.convert("RGB"))
                elif isinstance(frame, Image.Image):
                    array = np.asarray(frame.convert("RGB"))
                elif isinstance(frame, np.ndarray):
                    array = frame
                else:
                    raise TypeError(f"Unsupported frame type: {type(frame)!r}")
                if array.ndim != 3 or array.shape[-1] != 3:
                    raise ValueError(f"Expected an HxWx3 frame, got shape {array.shape}")
                # Match the training pipeline's ResizeImage(size=(448, 448)),
                # including PIL's default resize interpolation.
                resized = Image.fromarray(array).resize(recipe.IMAGE_SIZE)
                frames.append(np.asarray(resized).copy())
            loaded.append(frames)
        return loaded

    def build_prompt(self, views: Sequence[Sequence], instruction: str) -> dict[str, torch.Tensor]:
        """Build the memVLA user turn.

        Ported from ``PrepareMiniCPMMemVLAInputs.__call__``: per camera, a label,
        then each frame preceded by its timestamp tag, and finally the embodiment
        prompt. ``add_generation_prompt=True`` and no assistant turn, so the model
        answers with the sub-task.
        """
        loaded = self._load_views(views)
        content: list[dict] = []
        for view_index, frames in enumerate(loaded):
            content.append({"type": "text", "text": f"{recipe.CAMERA_LABELS[view_index]}:"})
            ages = recipe.VIEW_FRAME_AGES_SEC[view_index]
            for frame_index, frame in enumerate(frames):
                content.append({"type": "text", "text": recipe.frame_timestamp_tag(ages[frame_index])})
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": recipe.embodiment_prompt(instruction)})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        return {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _prepare_state(self, state: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape != (1, 1, recipe.UNIFIED_DIM):
            raise ValueError(
                f"state must have shape ({recipe.UNIFIED_DIM},), (1, {recipe.UNIFIED_DIM}), "
                f"or (1, 1, {recipe.UNIFIED_DIM}); got {tuple(state.shape)}"
            )
        return state

    @torch.inference_mode()
    def _generate_subtask(self, vlm_inputs: dict[str, torch.Tensor]) -> list[int]:
        """Greedy-decode the sub-task answer, one full forward pass per token.

        The language model is a linear-attention hybrid whose cache carries a
        conv state rather than standard ``past_key_values``, so incremental
        decoding is not available; only the vision encoding is memoized.
        """
        input_ids = vlm_inputs["input_ids"]
        attention_mask = vlm_inputs["attention_mask"]
        generated: list[int] = []

        # Match the action path's autocast(vlm_dtype) without forcing output_hidden_states
        # (generation only needs the last-token logits, so hidden states would be wasted
        # memory over ~24 steps on the 62-frame sequence).
        vlm_dtype = getattr(self.model, "vlm_dtype", None)
        autocast = torch.autocast("cuda", dtype=vlm_dtype, enabled=torch.cuda.is_available() and vlm_dtype is not None)
        cache = _cache_vision_encoding(self.model.vlm) if self.cache_vision else contextlib.nullcontext()
        with cache, autocast:
            for _ in range(self.max_subtask_tokens):
                outputs = self.model.vlm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=vlm_inputs["pixel_values"],
                    target_sizes=vlm_inputs["target_sizes"],
                    use_cache=False,
                    logits_to_keep=1,
                )
                next_token = outputs.logits[:, -1].argmax(dim=-1)
                generated.append(int(next_token[0]))
                if generated[-1] == self.im_end_id:
                    break
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token[:, None])], dim=1)
        return generated

    @torch.inference_mode()
    def predict(
        self,
        views: Sequence[Sequence],
        instruction: str,
        state: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        embodiment_id: int = recipe.EMBODIMENT_ID,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, str]:
        """Return ``(actions (30, 80) on CPU, sub-task text)``."""
        if not 0 <= embodiment_id < self.model.action_head.max_num_embodiments:
            raise ValueError(
                f"embodiment_id must be in [0, {self.model.action_head.max_num_embodiments - 1}]"
            )
        if state is None:
            state = torch.zeros(recipe.UNIFIED_DIM)
        state_tensor = self._prepare_state(state)
        embodiment = torch.tensor([embodiment_id], dtype=torch.long, device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        vlm_inputs = self.build_prompt(views, instruction)
        generated = self._generate_subtask(vlm_inputs)

        # Condition the action head on the prompt plus the generated sub-task.
        # predict_action takes the hidden states of the whole sequence, which is
        # the condition_span="full" training recipe.
        answer = torch.tensor([generated], dtype=vlm_inputs["input_ids"].dtype, device=self.device)
        conditioned = dict(vlm_inputs)
        conditioned["input_ids"] = torch.cat([vlm_inputs["input_ids"], answer], dim=1)
        conditioned["attention_mask"] = torch.cat(
            [vlm_inputs["attention_mask"], torch.ones_like(answer)], dim=1
        )
        actions = self.model.predict_action(
            state=state_tensor,
            embodiment_id=embodiment,
            **conditioned,
        )

        subtask_ids = generated[:-1] if generated and generated[-1] == self.im_end_id else generated
        subtask = self.processor.tokenizer.decode(subtask_ids, skip_special_tokens=True).strip()
        return actions[0].float().cpu(), subtask
