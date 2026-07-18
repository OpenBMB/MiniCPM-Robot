#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from contextlib import nullcontext
import os
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from transformers import AutoImageProcessor, AutoModel
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def _resize_rgb_pil(img: Image.Image, image_size: int) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    return img


def _pil_to_chw_uint8_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr)
    return tensor.permute(2, 0, 1).contiguous()


def _uint8_batch_to_float01(batch_uint8: torch.Tensor) -> torch.Tensor:
    return batch_uint8.to(dtype=torch.float32).div_(255.0)


def _sqrt_int(n: int) -> int:
    r = int(round(math.sqrt(n)))
    if r * r != n:
        raise ValueError(f"{n} is not a perfect square; can't form a square grid")
    return r


def _list_image_files(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def load_images_by_view_from_dir(root_dir: str) -> List[List[Image.Image]]:
    """Load images organized as views from a directory.

    Layouts supported:
      - root_dir/frames/seed_XXX/<scene>/<camera>/*.{jpg,png,...}  -> multi-view TrackVLA
      - root_dir/<view_name>/*.{jpg,png,...}                       -> each subdir is a view
      - root_dir/*.{jpg,png,...}                                   -> single view

    Selection rules for frames layout:
      - Pick seed from env TRACKVLA_SEED (number or folder), else latest numeric seed
      - Pick scene from env TRACKVLA_SCENE, else the first scene alphabetically
      - Optionally limit frames via env TRACKVLA_MAX_T (int) and stride via TRACKVLA_STEP (int)

    Ensures all views have the same T by truncating to the minimum length.
    """
    root = Path(root_dir)
    assert root.exists() and root.is_dir(), f"Directory not found: {root_dir}"

    # Branch 1: TrackVLA frames layout
    frames_dir = root / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        # Choose seed
        seed_env = os.getenv("TRACKVLA_SEED", "").strip()
        seed_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        assert len(seed_dirs) > 0, f"No seed_* directories under {frames_dir}"

        def _seed_order(p: Path) -> Tuple[int, str]:
            try:
                return (int(p.name.split("_")[1]), p.name)
            except Exception:
                return (-(1 << 30), p.name)

        if seed_env:
            # Allow number or full folder name
            if seed_env.isdigit():
                seed_choice = f"seed_{seed_env}"
            else:
                seed_choice = seed_env
            seed_path = frames_dir / seed_choice
            assert seed_path.exists(), f"Seed not found: {seed_choice}"
        else:
            seed_path = sorted(seed_dirs, key=_seed_order)[-1]  # latest numeric seed

        # Choose scene
        scene_env = os.getenv("TRACKVLA_SCENE", "").strip()
        scene_dirs = sorted([p for p in seed_path.iterdir() if p.is_dir()])
        assert len(scene_dirs) > 0, f"No scene directories under {seed_path}"
        if scene_env:
            scene_path = seed_path / scene_env
            assert scene_path.exists(), f"Scene not found: {scene_env}"
        else:
            scene_path = scene_dirs[0]

        # Views = subdirectories within scene (e.g., camera indices: 0,1,2,...)
        view_dirs = sorted([p for p in scene_path.iterdir() if p.is_dir()])
        images_by_view: List[List[Image.Image]] = []

        if len(view_dirs) == 0:
            # Fallback: images directly under scene_path
            files = _list_image_files(scene_path)
            assert len(files) > 0, f"No images found in {scene_path}"
            imgs = [Image.open(str(p)).convert("RGB") for p in files]
            images_by_view = [imgs]
        else:
            for vd in view_dirs:
                files = _list_image_files(vd)
                if len(files) == 0:
                    continue
                imgs = [Image.open(str(p)).convert("RGB") for p in files]
                images_by_view.append(imgs)
            assert len(images_by_view) > 0, f"No images found under any camera dir of {scene_path}"

        # Align and optionally subsample
        min_T = min(len(imgs) for imgs in images_by_view)
        max_t_env = os.getenv("TRACKVLA_MAX_T")
        step_env = os.getenv("TRACKVLA_STEP")
        step = int(step_env) if (step_env and step_env.isdigit()) else 1
        if max_t_env and max_t_env.isdigit():
            min_T = min(min_T, int(max_t_env))
        indices = list(range(0, min_T, step))
        images_by_view = [[imgs[i] for i in indices] for imgs in images_by_view]
        return images_by_view

    # Branch 2: <root>/<view>/*
    view_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    images_by_view: List[List[Image.Image]] = []
    if len(view_dirs) > 0:
        for vd in view_dirs:
            files = _list_image_files(vd)
            if len(files) == 0:
                continue
            imgs = [Image.open(str(p)).convert("RGB") for p in files]
            images_by_view.append(imgs)
        assert len(images_by_view) > 0, f"No images found under any subdirectory of {root_dir}"
        # Align and optionally subsample
        min_T = min(len(imgs) for imgs in images_by_view)
        max_t_env = os.getenv("TRACKVLA_MAX_T")
        step_env = os.getenv("TRACKVLA_STEP")
        step = int(step_env) if (step_env and step_env.isdigit()) else 1
        if max_t_env and max_t_env.isdigit():
            min_T = min(min_T, int(max_t_env))
        indices = list(range(0, min_T, step))
        images_by_view = [[imgs[i] for i in indices] for imgs in images_by_view]
        return images_by_view

    # Branch 3: <root>/*
    files = _list_image_files(root)
    assert len(files) > 0, f"No images found in {root_dir}"
    imgs = [Image.open(str(p)).convert("RGB") for p in files]
    # Optionally subsample
    max_t_env = os.getenv("TRACKVLA_MAX_T")
    step_env = os.getenv("TRACKVLA_STEP")
    step = int(step_env) if (step_env and step_env.isdigit()) else 1
    if max_t_env and max_t_env.isdigit():
        imgs = imgs[:int(max_t_env)]
    imgs = imgs[::step] if step > 1 else imgs
    return [imgs]


# -----------------------------------------------------------------------------
# Grid Average Pooling (token-level)
# -----------------------------------------------------------------------------

def grid_pool_tokens(patch_tokens: torch.Tensor, Hp: int, Wp: int, out_tokens: int) -> torch.Tensor:
    """Average-pool patch tokens on a HxW grid down to out_tokens (must be square).

    Args:
        patch_tokens: (B, P, C)
        Hp, Wp: patch grid height/width s.t. P == Hp*Wp
        out_tokens: 64 or 4 (must be a square number)
    Returns:
        pooled: (B, out_tokens, C)
    """
    B, P, C = patch_tokens.shape
    assert P == Hp * Wp, f"P={P} must equal Hp*Wp={Hp*Wp}"
    s = _sqrt_int(out_tokens)
    # (B, P, C) -> (B, C, Hp, Wp)
    feat = patch_tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
    feat = F.adaptive_avg_pool2d(feat, output_size=(s, s))  # (B, C, s, s)
    pooled = feat.flatten(2).transpose(1, 2).contiguous()    # (B, s*s, C)
    return pooled


def adapt_siglip_grid(tokens: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None, out_hw: Tuple[int, int] = (24, 24)) -> torch.Tensor:
    """Convert SigLIP tokens to a target grid (default 24x24) via adaptive avg pool.

    Args:
        tokens: (B, P_s, C_s)
        grid_hw: (Hs, Ws) if known, otherwise inferred as sqrt(P_s)
        out_hw: target (H_out, W_out)
    Returns:
        (B, H_out*W_out, C_s)
    """
    B, P_s, C_s = tokens.shape
    if grid_hw is None:
        Hs = _sqrt_int(P_s)
        Ws = Hs
    else:
        Hs, Ws = grid_hw
        assert Hs * Ws == P_s
    feat = tokens.transpose(1, 2).contiguous().view(B, C_s, Hs, Ws)
    feat = F.adaptive_avg_pool2d(feat, output_size=out_hw)
    pooled = feat.flatten(2).transpose(1, 2).contiguous()  # (B, 24*24, C_s) if out_hw=(24,24)
    return pooled


# -----------------------------------------------------------------------------
# DINOv3 + SigLIP Feature Extractor & Cacher
# -----------------------------------------------------------------------------

@dataclass
class VisionCacheConfig:
    dino_model_name: str = None  # Will be set from env or default
    siglip_model_name: str = None  # Will be set from env or default
    image_size: int = 384          # enforce 384 for both towers
    batch_size: int = 24
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16  # for storage; compute still in fp32
    compute_dtype: torch.dtype = torch.float32
    vision_amp_dtype: Optional[torch.dtype] = None
    force_square_resize: bool = True    # ensure exact 384x384
    timing_sync: bool = True             # accurate per-stage timings; disable for lower live latency
    timing_mode: str = "accurate"        # accurate | fast | event

    def __post_init__(self):
        if self.dino_model_name is None:
            env_path = os.getenv("DINOV3_MODEL_PATH", "").strip()
            if env_path and os.path.exists(env_path):
                self.dino_model_name = env_path
            else:
                self.dino_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        if self.siglip_model_name is None:
            env_path = os.getenv("SIGLIP_MODEL_PATH", "").strip()
            if env_path and os.path.exists(env_path):
                self.siglip_model_name = env_path
            else:
                self.siglip_model_name = "google/siglip-so400m-patch14-384"


class VisionFeatureCacher(nn.Module):
    def __init__(self, cfg: VisionCacheConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # DINOv3
        self.dino_proc = AutoImageProcessor.from_pretrained(cfg.dino_model_name)
        self.dino = AutoModel.from_pretrained(cfg.dino_model_name)
        self.dino.eval().to(self.device)
        self.dino_patch = getattr(self.dino.config, 'patch_size', None)
        # Default to 0 registers when field is missing
        self.dino_regs = int(getattr(self.dino.config, 'num_register_tokens', 0) or 0)
        self.dino_hidden = getattr(self.dino.config, 'hidden_size', 384)
        # SigLIP (vision)
        self.siglip_proc = SiglipImageProcessor.from_pretrained(cfg.siglip_model_name)
        self.siglip = SiglipVisionModel.from_pretrained(cfg.siglip_model_name)
        self.siglip.eval().to(self.device)
        self.siglip_hidden = getattr(self.siglip.config, 'hidden_size', 1152)
        self._dino_mean = torch.tensor(
            list(getattr(self.dino_proc, "image_mean", [0.485, 0.456, 0.406])),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1, 1, 1)
        self._dino_std = torch.tensor(
            list(getattr(self.dino_proc, "image_std", [0.229, 0.224, 0.225])),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1, 1, 1)
        self._siglip_mean = torch.tensor(
            list(getattr(self.siglip_proc, "image_mean", [0.5, 0.5, 0.5])),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1, 1)
        self._siglip_std = torch.tensor(
            list(getattr(self.siglip_proc, "image_std", [0.5, 0.5, 0.5])),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1, 1)

    def _vision_autocast(self):
        if self.device.type != "cuda" or self.cfg.vision_amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.cfg.vision_amp_dtype)

    def _timing_event_enabled(self) -> bool:
        return str(getattr(self.cfg, "timing_mode", "accurate")).strip().lower() == "event" and self.device.type == "cuda"

    def _record_cuda_event(self) -> Optional[torch.cuda.Event]:
        if not self._timing_event_enabled():
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(self.device))
        return ev

    @staticmethod
    def _append_cuda_event(
        events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]],
        name: str,
        start_ev: Optional[torch.cuda.Event],
        end_ev: Optional[torch.cuda.Event],
    ) -> None:
        if start_ev is not None and end_ev is not None:
            events.append((name, start_ev, end_ev))

    # ---------------------------------------
    # Encoding helpers
    # ---------------------------------------
    def _prepare_shared_batch(self, pil_list: List[Image.Image]) -> torch.Tensor:
        if self.cfg.force_square_resize:
            pil_list = [_resize_rgb_pil(im, self.cfg.image_size) for im in pil_list]
        else:
            pil_list = [im.convert("RGB") if im.mode != "RGB" else im for im in pil_list]

        batch_uint8 = torch.stack([_pil_to_chw_uint8_tensor(im) for im in pil_list], dim=0)
        if self.device.type == "cuda":
            batch_uint8 = batch_uint8.pin_memory()
        batch = _uint8_batch_to_float01(batch_uint8.to(self.device, non_blocking=(self.device.type == "cuda")))
        return batch

    def _preprocess_pils(self, pil_list: List[Image.Image], which: str) -> Dict[str, torch.Tensor]:
        batch = self._prepare_shared_batch(pil_list)
        if which == 'dino':
            pixel_values = (batch - self._dino_mean) / self._dino_std
        else:
            pixel_values = (batch - self._siglip_mean.unsqueeze(0)) / self._siglip_std.unsqueeze(0)
        return {"pixel_values": pixel_values}

    @torch.inference_mode()
    def _encode_dino(self, pil_list: List[Image.Image], shared_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int, int]:
        if shared_batch is None:
            inputs = self._preprocess_pils(pil_list, which='dino')
        else:
            inputs = {"pixel_values": (shared_batch - self._dino_mean) / self._dino_std}
        with self._vision_autocast():
            out = self.dino(**inputs)
        tok = out.last_hidden_state  # (B, 1+R+P, C)
        if not getattr(self.dino, "patch_tokens_only", False):
            tok = tok[:, 1 + self.dino_regs:, :]  # drop CLS + registers -> (B, P, C)
        P = tok.size(1)
        Hp = _sqrt_int(P)
        Wp = Hp
        return tok, Hp, Wp

    @torch.inference_mode()
    def _encode_siglip(self, pil_list: List[Image.Image], out_hw: Tuple[int, int], shared_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if shared_batch is None:
            inputs = self._preprocess_pils(pil_list, which='siglip')
        else:
            inputs = {"pixel_values": (shared_batch - self._siglip_mean.unsqueeze(0)) / self._siglip_std.unsqueeze(0)}
        with self._vision_autocast():
            out = self.siglip(**inputs)
        tok = out.last_hidden_state  # (B, S, C)
        # SigLIP may include a CLS token; infer and drop if present
        S = tok.size(1)
        Hs = int(round(math.sqrt(S)))
        if Hs * Hs == S - 1:
            tok = tok[:, 1:, :]  # drop CLS -> (B, Hs*Hs, C)
            S = tok.size(1)
            Hs = int(round(math.sqrt(S)))
        assert Hs * Hs == S, f"SigLIP tokens not square (S={S})"
        # Pool SigLIP grid (Hs×Hs) to target DINO grid (Hp×Wp)
        tok = adapt_siglip_grid(tok, grid_hw=(Hs, Hs), out_hw=out_hw)
        return tok

    @torch.inference_mode()
    def encode_frame_tokens_from_rgb_with_timing(
        self, rgb_np
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Encode a single RGB numpy frame and return pooled coarse/fine tokens."""
        timing: Dict[str, float] = {}
        cuda_events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        t0 = time.perf_counter()
        ev0 = self._record_cuda_event()
        rgb_u8 = np.asarray(rgb_np, dtype=np.uint8)
        if not rgb_u8.flags.c_contiguous:
            rgb_u8 = np.ascontiguousarray(rgb_u8)
        pil = _resize_rgb_pil(Image.fromarray(rgb_u8, mode="RGB"), self.cfg.image_size)
        shared_batch = self._prepare_shared_batch([pil])
        ev1 = self._record_cuda_event()
        if self.cfg.timing_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["vision_preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        self._append_cuda_event(cuda_events, "vision_preprocess_ms", ev0, ev1)

        t1 = time.perf_counter()
        ev2 = self._record_cuda_event()
        tok_dino, hp, wp = self._encode_dino([pil], shared_batch=shared_batch)
        ev3 = self._record_cuda_event()
        if self.cfg.timing_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["vision_dino_ms"] = (time.perf_counter() - t1) * 1000.0
        self._append_cuda_event(cuda_events, "vision_dino_ms", ev2, ev3)

        t2 = time.perf_counter()
        ev4 = self._record_cuda_event()
        tok_sigl = self._encode_siglip([pil], out_hw=(hp, wp), shared_batch=shared_batch)
        ev5 = self._record_cuda_event()
        if self.cfg.timing_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["vision_siglip_ms"] = (time.perf_counter() - t2) * 1000.0
        self._append_cuda_event(cuda_events, "vision_siglip_ms", ev4, ev5)

        t3 = time.perf_counter()
        ev6 = self._record_cuda_event()
        vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(vt_cat, hp, wp, out_tokens=64)[0]
        vcoarse = grid_pool_tokens(vt_cat, hp, wp, out_tokens=4)[0]
        if self.cfg.compute_dtype is not None:
            vfine = vfine.to(self.cfg.compute_dtype)
            vcoarse = vcoarse.to(self.cfg.compute_dtype)
        ev7 = self._record_cuda_event()
        if self.cfg.timing_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["vision_pool_ms"] = (time.perf_counter() - t3) * 1000.0
        self._append_cuda_event(cuda_events, "vision_pool_ms", ev6, ev7)
        if cuda_events:
            timing["_cuda_events"] = cuda_events

        return vcoarse, vfine, timing

    # ---------------------------------------
    # Public API
    # ---------------------------------------
    @torch.inference_mode()
    def encode_views(
        self,
        images_by_view: List[List[Image.Image]],
        out_dir: str,
        seq_id: Optional[str] = "seq0001",
        filenames_by_view: Optional[List[List[str]]] = None,
    ) -> None:
        """Cache a full [V][T] image set to per-time .pt files with pooled tokens.

        Args:
            images_by_view: nested list shaped [V][T] of PIL Images
            out_dir: directory to place the cache
            seq_id: subfolder name for this sequence
        """
        V = len(images_by_view)
        assert V > 0, "Need at least one view"
        T = len(images_by_view[0])
        for v in range(V):
            assert len(images_by_view[v]) == T, "All views must have same #frames"

        Hp: Optional[int] = None
        Wp: Optional[int] = None
        P: Optional[int] = None

        # If seq_id is empty/None, write directly into out_dir
        seq_dir = out_dir if (seq_id is None or seq_id == "") else os.path.join(out_dir, seq_id)
        ensure_dir(seq_dir)

        # Flatten to [V*T] for efficient batching
        flat_images: List[Image.Image] = []
        index: List[Dict[str, int]] = []
        for t in range(T):
            for v in range(V):
                flat_images.append(images_by_view[v][t])
                index.append({"t": t, "v": v})

        # Process in mini-batches
        Bsz = self.cfg.batch_size
        feats_dino: List[torch.Tensor] = []  # (b, 576, C_dino)
        feats_sigl: List[torch.Tensor] = []  # (b, 576, C_siglip)
        for start in range(0, len(flat_images), Bsz):
            end = min(len(flat_images), start + Bsz)
            batch_pils = flat_images[start:end]
            tok_dino, Hp_chk, Wp_chk = self._encode_dino(batch_pils)
            if Hp is None:
                Hp, Wp, P = Hp_chk, Wp_chk, tok_dino.size(1)
            else:
                assert Hp_chk == Hp and Wp_chk == Wp and tok_dino.size(1) == P
            tok_sigl = self._encode_siglip(batch_pils, out_hw=(Hp, Wp))
            assert tok_sigl.size(1) == P
            feats_dino.append(tok_dino.cpu())
            feats_sigl.append(tok_sigl.cpu())

        all_dino = torch.cat(feats_dino, dim=0)      # (V*T, 576, C_dino)
        all_sigl = torch.cat(feats_sigl, dim=0)      # (V*T, 576, C_siglip)
        C_dino = all_dino.size(-1)
        C_sigl = all_sigl.size(-1)
        C_total = C_dino + C_sigl

        # Reshape back to (V, T, P, C)
        all_dino = all_dino.view(V, T, P, C_dino)
        all_sigl = all_sigl.view(V, T, P, C_sigl)

        # Write per-timestep cache (concat channels then GridPool)
        meta = {
            "seq_id": seq_id,
            "V": V,
            "T": T,
            "P": P,
            "Hp": Hp,
            "Wp": Wp,
            "image_size": self.cfg.image_size,
            "dino": {
                "model_name": self.cfg.dino_model_name,
                "patch_size": self.dino_patch,
                "num_register_tokens": self.dino_regs,
                "hidden_size": self.dino_hidden,
            },
            "siglip": {
                "model_name": self.cfg.siglip_model_name,
                "hidden_size": self.siglip_hidden,
                "pooled_to_grid": [Hp, Wp],
            },
            "concat": {
                "hidden_size_total": C_total
            }
        }
        with open(os.path.join(seq_dir, "index.json"), "w") as f:
            json.dump(meta, f, indent=2)

        save_per_view = filenames_by_view is not None
        for t in range(T):
            Vt_dino = all_dino[:, t, :, :]  # (V, P, C_dino)
            Vt_sigl = all_sigl[:, t, :, :]  # (V, P, C_siglip)
            Vt_cat = torch.cat([Vt_dino, Vt_sigl], dim=-1)  # (V, P, C_total)

            # GridPool on concatenated channels (paper):
            Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)   # (V, 64, C_total)
            Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)  # (V, 4, C_total)

            if save_per_view:
                # Save two files per view using original filename base
                for v in range(V):
                    base_name = os.path.splitext(os.path.basename(filenames_by_view[v][t]))[0]
                    vf_path = os.path.join(seq_dir, f"{base_name}_vfine.pt")
                    vc_path = os.path.join(seq_dir, f"{base_name}_vcoarse.pt")
                    torch.save(Vfine[v].to(self.cfg.dtype), vf_path)
                    torch.save(Vcoarse[v].to(self.cfg.dtype), vc_path)
            else:
                St = {
                    "V_dino": Vt_dino.to(self.cfg.dtype),
                    "V_siglip": Vt_sigl.to(self.cfg.dtype),
                    "V": Vt_cat.to(self.cfg.dtype),
                    "Vfine": Vfine.to(self.cfg.dtype),
                    "Vcoarse": Vcoarse.to(self.cfg.dtype),
                    "meta": {"t": t}
                }
                torch.save(St, os.path.join(seq_dir, f"t{t:05d}.pt"))

        print(f"Cached sequence to: {seq_dir}  (V={V}, T={T}, P={P}, C_total={C_total})")


# -----------------------------------------------------------------------------
# Optional: Cross-Modality Projector P(·) to LLM space
# -----------------------------------------------------------------------------

class CrossModalityProjector(nn.Module):
    """Two-layer MLP projector P(·) mapping vision features (C_total) -> LLM dim (D)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Load frames from data/track and cache grid-pooled tokens
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data", "track")
    out_root = os.path.join(script_dir, "vision_cache")

    frames_root = os.path.join(data_dir, "frames")
    if os.path.isdir(frames_root):
        # Mirror frames/seed_X/scene into vision_cache/frames/seed_X/scene
        seeds = sorted([d for d in os.listdir(frames_root) if d.startswith("seed_")])
        seed_env = os.getenv("TRACKVLA_SEED", "").strip()
        if seed_env:
            if seed_env.isdigit():
                seed_choice = f"seed_{seed_env}"
            else:
                seed_choice = seed_env
            seeds = [s for s in seeds if s == seed_choice]
        if len(seeds) == 0:
            raise RuntimeError(f"No seed_* directories under {frames_root}")

        cfg = VisionCacheConfig(image_size=384, batch_size=8)
        cacher = VisionFeatureCacher(cfg)

        for seed in seeds:
            seed_path = os.path.join(frames_root, seed)
            scenes = sorted([s for s in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, s))])
            scene_env = os.getenv("TRACKVLA_SCENE", "").strip()
            if scene_env:
                scenes = [s for s in scenes if s == scene_env]
            for scene in scenes:
                scene_path = os.path.join(seed_path, scene)
                # Each camera dir becomes its own cache subfolder
                cam_dirs = sorted([c for c in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, c))])
                if len(cam_dirs) == 0:
                    continue
                for cam in cam_dirs:
                    cdir = os.path.join(scene_path, cam)
                    files = sorted([os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
                    if len(files) == 0:
                        continue
                    imgs = [Image.open(fp).convert("RGB") for fp in files]
                    # Optionally subsample
                    max_t_env = os.getenv("TRACKVLA_MAX_T")
                    step_env = os.getenv("TRACKVLA_STEP")
                    step = int(step_env) if (step_env and step_env.isdigit()) else 1
                    if max_t_env and max_t_env.isdigit():
                        files = files[:int(max_t_env)]
                        imgs = imgs[:int(max_t_env)]
                    if step > 1:
                        files = files[::step]
                        imgs = imgs[::step]

                    save_dir = os.path.join(out_root, "frames", seed, scene, cam)
                    cacher.encode_views([imgs], out_dir=save_dir, seq_id=None, filenames_by_view=[files])
                    print(f"Done: {seed}/{scene}/{cam} -> {save_dir}")
    else:
        # Fallback: previous single-folder behavior
        images_by_view: List[List[Image.Image]] = load_images_by_view_from_dir(data_dir)
        cfg = VisionCacheConfig(image_size=384, batch_size=8)
        cacher = VisionFeatureCacher(cfg)
        seq_id = "track_seq"
        cacher.encode_views(images_by_view, out_dir=out_root, seq_id=seq_id)

        # Optional: project one timestep to LLM dim=256
        sample_path = os.path.join(out_root, seq_id, "t00000.pt")
        if os.path.exists(sample_path):
            sample = torch.load(sample_path)
            Vfine = sample["Vfine"]  # (V, 64, C_total)
            projector = CrossModalityProjector(in_dim=Vfine.shape[-1], out_dim=256)
            EVfine = projector(Vfine.float())  # (V, 64, 256)
            print("Projected tokens shape:", tuple(EVfine.shape))
