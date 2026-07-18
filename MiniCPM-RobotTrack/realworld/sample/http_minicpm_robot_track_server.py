#!/usr/bin/env python3
import argparse
import inspect
import io
import json
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from multiprocessing import shared_memory
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from PIL import Image
from transformers import AutoModel, AutoTokenizer

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trt_artifacts"))
DEFAULT_DINO_ENGINE = os.path.join(ARTIFACTS_DIR, "dino_patch_target_fp16.engine")
DEFAULT_SIGLIP_ENGINE = os.path.join(ARTIFACTS_DIR, "siglip_pooled_target_maxn_fp16.engine")

def _discover_minicpm_robot_track_root() -> str:
    candidates = [
        os.environ.get("MINICPM_ROBOT_TRACK_ROOT", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(
            os.path.join(candidate, "minicpm_robot_track", "vision_cache.py")
        ):
            return candidate
    raise FileNotFoundError(
        "Cannot find MiniCPM-RobotTrack root; set MINICPM_ROBOT_TRACK_ROOT"
    )


PROJECT_ROOT = _discover_minicpm_robot_track_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from minicpm_robot_track.vision_cache import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens
from waypoint_controller import WaypointVelocityController, WaypointVelocityControllerConfig

def _is_hf_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    cfg = os.path.join(path, "config.json")
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return os.path.isfile(cfg) and any(
        os.path.isfile(os.path.join(path, filename)) for filename in weight_files
    )


class MiniCPMRobotTrackRealworldAgent:
    """Inference wrapper shared by HTTP service and offline tools."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.history = int(args.history)
        self.image_size = int(args.image_size)
        self.control_dt = float(args.control_dt)
        self.max_vx = float(args.max_vx)
        self.max_vy = float(args.max_vy)
        self.max_wz = float(args.max_wz)
        self.default_instruction = str(args.instruction)
        self._coarse_tidx_template: Optional[torch.Tensor] = None
        self._fine_tidx_template: Optional[torch.Tensor] = None
        self._coarse_tidx_shape: Optional[Tuple[int, int]] = None
        self._fine_tidx_len: Optional[int] = None
        self._policy_graph_state: Optional[Dict[str, Any]] = None
        self._policy_graph_disabled = False
        self._coarse_hist_proj_tokens: deque = deque(maxlen=self.history)
        self._prompt_token_cache: Dict[
            Tuple[str, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

        req_device = str(args.device)
        if req_device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(req_device)
        else:
            self.device = torch.device("cpu")

        self._vision_cache: Optional[VisionFeatureCacher] = None
        self._coarse_hist_tokens: deque = deque(maxlen=self.history)
        self.tokenizer = None
        self.model = self._load_model()
        self.waypoint_controller = self._build_waypoint_controller()

    def _timing_mode(self) -> str:
        return str(getattr(self.args, "timing_mode", "accurate")).strip().lower()

    def _timing_sync_enabled(self) -> bool:
        return self._timing_mode() == "accurate"

    def _timing_event_enabled(self) -> bool:
        return self._timing_mode() == "event" and self.device.type == "cuda"

    def _maybe_cuda_sync(self) -> None:
        if self._timing_sync_enabled() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record_cuda_event(self) -> Optional[torch.cuda.Event]:
        if not self._timing_event_enabled():
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(self.device))
        return ev

    @staticmethod
    def _resolve_cuda_events(timing: Dict[str, Any]) -> None:
        events = timing.pop("_cuda_events", None)
        if not events:
            return
        for name, start_ev, end_ev in events:
            try:
                timing[name] = float(start_ev.elapsed_time(end_ev))
            except Exception:
                pass

    def _get_tidx_templates(self, coarse_per_frame: int, fine_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_shape = (int(self.history), int(coarse_per_frame))
        if self._coarse_tidx_template is None or self._coarse_tidx_shape != coarse_shape:
            coarse = torch.arange(self.history, dtype=torch.long, device=self.device).repeat_interleave(coarse_per_frame)
            self._coarse_tidx_template = coarse.unsqueeze(0)
            self._coarse_tidx_shape = coarse_shape
        if self._fine_tidx_template is None or self._fine_tidx_len != int(fine_tokens):
            self._fine_tidx_template = torch.full(
                (1, int(fine_tokens)),
                self.history,
                dtype=torch.long,
                device=self.device,
            )
            self._fine_tidx_len = int(fine_tokens)
        return self._coarse_tidx_template, self._fine_tidx_template

    def _build_waypoint_controller(self) -> WaypointVelocityController:
        """Construct the waypoint→velocity controller from CLI args.

        The direct wp_idx=1 defaults match the validated on-device runtime.
        """
        a = self.args
        cfg = WaypointVelocityControllerConfig(
            control_dt=float(a.control_dt),
            max_vx=float(a.max_vx),
            max_vy=float(a.max_vy),
            max_wz=float(a.max_wz),
            disable_lateral=bool(getattr(a, "disable_lateral", True)),
            controller_mode=str(getattr(a, "velocity_controller", "direct")),
            direct_wp_idx=int(getattr(a, "controller_direct_wp_idx", 1)),
            lookahead_idx=int(getattr(a, "controller_lookahead_idx", 2)),
            feedforward_points=int(getattr(a, "controller_feedforward_points", 3)),
            ff_blend=float(getattr(a, "controller_ff_blend", 0.65)),
            kx=float(getattr(a, "controller_kx", 1.2)),
            ky=float(getattr(a, "controller_ky", 1.0)),
            ktheta=float(getattr(a, "controller_ktheta", 1.4)),
            curvature_gain=float(getattr(a, "controller_curvature_gain", 0.35)),
            min_turn_speed=float(getattr(a, "controller_min_turn_speed", 0.08)),
            allow_reverse=bool(getattr(a, "allow_reverse", False)),
            turn_in_place_factor=float(getattr(a, "turn_in_place_factor", 0.0)),
            min_vx_factor=float(getattr(a, "min_vx_factor", 0.2)),
        )
        print(
            f"[server] waypoint controller mode={cfg.controller_mode} "
            f"wp_idx={cfg.direct_wp_idx} max_v=({cfg.max_vx},{cfg.max_vy},{cfg.max_wz})"
        )
        return WaypointVelocityController(cfg)

    def _resolve_model_path(self) -> str:
        model_dir = str(self.args.model_dir or "").strip()
        if not _is_hf_dir(model_dir):
            raise FileNotFoundError(
                f"Invalid --model_dir: {model_dir}. Need a complete Hugging Face "
                "snapshot containing config.json and model weights."
            )
        return model_dir

    def _load_model(self):
        model_path = self._resolve_model_path()
        print(f"[server] loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = model.to(self.device).eval()
        return model

    def _tokenize_prompt(
        self, prompt: list[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = tuple(prompt)
        cached = self._prompt_token_cache.get(key)
        if cached is not None:
            return cached
        if self.tokenizer is None:
            raise RuntimeError("model tokenizer is not initialized")
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.model.config.max_text_tokens),
        )
        tensors = (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )
        self._prompt_token_cache[key] = tensors
        return tensors

    def _ensure_vision_cache(self) -> VisionFeatureCacher:
        if self._vision_cache is None:
            amp_dtype = self._resolve_amp_dtype(getattr(self.args, "vision_amp", "none"))
            cfg_kwargs = {
                "image_size": self.image_size,
                "batch_size": 1,
                "device": ("cuda" if self.device.type == "cuda" else "cpu"),
                "compute_dtype": torch.float32,
                "vision_amp_dtype": amp_dtype,
                "timing_sync": self._timing_sync_enabled(),
                "timing_mode": self._timing_mode(),
            }
            cfg_params = inspect.signature(VisionCacheConfig).parameters
            cfg = VisionCacheConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_params})
            self._vision_cache = VisionFeatureCacher(cfg)
            self._vision_cache.eval()
            print(f"[server] vision_amp_dtype={amp_dtype}")
            self._maybe_install_dino_trt(self._vision_cache)
            self._maybe_install_siglip_trt(self._vision_cache)
        return self._vision_cache

    def _maybe_install_dino_trt(self, cacher: VisionFeatureCacher) -> None:
        backend = str(getattr(self.args, "dino_backend", "torch")).lower()
        if backend == "torch":
            return
        if backend != "trt_direct":
            raise ValueError(f"Unsupported --dino_backend={backend!r}")

        from dino_trt_direct import DinoTrtDirect

        engine = str(getattr(self.args, "dino_trt_engine", ""))
        output_tokens = int(getattr(self.args, "dino_trt_output_tokens", 576))
        proxy = DinoTrtDirect(
            engine_path=engine or DEFAULT_DINO_ENGINE,
            output_tokens=output_tokens,
        )
        try:
            del cacher.dino
        except Exception:
            pass
        cacher.dino = proxy
        cacher.dino_hidden = 384
        cacher.dino_patch = 16
        cacher.dino_regs = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[server] DINO backend = TRT FP16 (in-process TensorRT)")

    def _maybe_install_siglip_trt(self, cacher: VisionFeatureCacher) -> None:
        backend = str(getattr(self.args, "siglip_backend", "torch")).lower()
        if backend == "torch":
            return

        engine = str(getattr(self.args, "siglip_trt_engine", ""))
        if backend == "trt_direct":
            from siglip_trt_direct import SigLipTrtDirect

            output_tokens = int(getattr(self.args, "siglip_trt_output_tokens", 729))
            proxy = SigLipTrtDirect(
                engine_path=engine or DEFAULT_SIGLIP_ENGINE,
                output_tokens=output_tokens,
            )
            detail = "in-process TensorRT"
        else:
            raise ValueError(f"Unsupported --siglip_backend={backend!r}")
        try:
            del cacher.siglip
        except Exception:
            pass
        cacher.siglip = proxy
        cacher.siglip_hidden = 1152
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[server] SigLIP backend = TRT FP16 ({detail})")

    @staticmethod
    def _resolve_amp_dtype(name: str) -> Optional[torch.dtype]:
        name = (name or "").lower().strip()
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        return None

    def _encode_frame_tokens(self, rgb_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._ensure_vision_cache()
        if hasattr(enc, "encode_frame_tokens_from_rgb_with_timing"):
            vcoarse, vfine, timing = enc.encode_frame_tokens_from_rgb_with_timing(rgb_np)
            return vcoarse, vfine, timing

        timing: Dict[str, float] = {}
        t0 = time.perf_counter()
        rgb_u8 = np.asarray(rgb_np, dtype=np.uint8)
        if not rgb_u8.flags.c_contiguous:
            rgb_u8 = np.ascontiguousarray(rgb_u8)
        pil = Image.fromarray(rgb_u8, mode="RGB")
        if getattr(enc.cfg, "force_square_resize", True) and pil.size != (self.image_size, self.image_size):
            pil = pil.resize((self.image_size, self.image_size), Image.BICUBIC)
        self._maybe_cuda_sync()
        timing["vision_preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        tok_dino, hp, wp = enc._encode_dino([pil])
        self._maybe_cuda_sync()
        timing["vision_dino_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        tok_sigl = enc._encode_siglip([pil], out_hw=(hp, wp))
        self._maybe_cuda_sync()
        timing["vision_siglip_ms"] = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(vt_cat, hp, wp, out_tokens=64)[0].float()
        vcoarse = grid_pool_tokens(vt_cat, hp, wp, out_tokens=4)[0].float()
        self._maybe_cuda_sync()
        timing["vision_pool_ms"] = (time.perf_counter() - t3) * 1000.0
        return vcoarse, vfine, timing

    def _policy_cuda_graph_enabled(self) -> bool:
        return (
            bool(getattr(self.args, "planner_cuda_graph", False))
            and self.device.type == "cuda"
            and not self._policy_graph_disabled
        )

    def _policy_forward_normal(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        prompt: list[str],
    ) -> torch.Tensor:
        input_ids, attention_mask = self._tokenize_prompt(prompt)

        def forward() -> torch.Tensor:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                coarse_tokens=coarse_tokens,
                coarse_time_indices=coarse_tidx,
                fine_tokens=fine_tokens,
                fine_time_indices=fine_tidx,
            )
            trajectories = getattr(outputs, "trajectories", None)
            if trajectories is None:
                raise RuntimeError("MiniCPM-RobotTrack output has no trajectories")
            return trajectories

        planner_amp = self._resolve_amp_dtype(getattr(self.args, "planner_amp", "none"))
        if self.device.type == "cuda" and planner_amp is not None:
            with torch.autocast(device_type="cuda", dtype=planner_amp):
                return forward()
        return forward()

    @staticmethod
    def _graph_tensor_signature(x: torch.Tensor) -> Tuple[Tuple[int, ...], str, str]:
        return (tuple(int(v) for v in x.shape), str(x.dtype), str(x.device))

    def _policy_graph_matches(
        self,
        state: Dict[str, Any],
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        prompt: list[str],
    ) -> bool:
        sig = (
            self._graph_tensor_signature(coarse_tokens),
            self._graph_tensor_signature(coarse_tidx),
            self._graph_tensor_signature(fine_tokens),
            self._graph_tensor_signature(fine_tidx),
        )
        return state.get("signature") == sig and state.get("prompt") == tuple(prompt)

    def _build_policy_cuda_graph(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        prompt: list[str],
    ) -> Dict[str, Any]:
        static_coarse = torch.empty_like(coarse_tokens)
        static_coarse_tidx = torch.empty_like(coarse_tidx)
        static_fine = torch.empty_like(fine_tokens)
        static_fine_tidx = torch.empty_like(fine_tidx)
        static_coarse.copy_(coarse_tokens)
        static_coarse_tidx.copy_(coarse_tidx)
        static_fine.copy_(fine_tokens)
        static_fine_tidx.copy_(fine_tidx)

        # Populate text-embedding cache and allocator pools before capture.
        for _ in range(max(1, int(getattr(self.args, "planner_cuda_graph_warmup", 3)))):
            _ = self._policy_forward_normal(
                static_coarse,
                static_coarse_tidx,
                static_fine,
                static_fine_tidx,
                prompt,
            )
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._policy_forward_normal(
                static_coarse,
                static_coarse_tidx,
                static_fine,
                static_fine_tidx,
                prompt,
            )
        torch.cuda.synchronize(self.device)
        sig = (
            self._graph_tensor_signature(coarse_tokens),
            self._graph_tensor_signature(coarse_tidx),
            self._graph_tensor_signature(fine_tokens),
            self._graph_tensor_signature(fine_tidx),
        )
        print(
            "[server] planner CUDA Graph captured "
            f"coarse={tuple(coarse_tokens.shape)} fine={tuple(fine_tokens.shape)} prompt={prompt!r}",
            flush=True,
        )
        return {
            "graph": graph,
            "coarse": static_coarse,
            "coarse_tidx": static_coarse_tidx,
            "fine": static_fine,
            "fine_tidx": static_fine_tidx,
            "out": static_out,
            "signature": sig,
            "prompt": tuple(prompt),
        }

    def _policy_forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        prompt: list[str],
    ) -> torch.Tensor:
        if not self._policy_cuda_graph_enabled():
            return self._policy_forward_normal(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, prompt)
        try:
            state = self._policy_graph_state
            if state is None or not self._policy_graph_matches(
                state, coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, prompt
            ):
                state = self._build_policy_cuda_graph(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, prompt)
                self._policy_graph_state = state
            state["coarse"].copy_(coarse_tokens)
            state["coarse_tidx"].copy_(coarse_tidx)
            state["fine"].copy_(fine_tokens)
            state["fine_tidx"].copy_(fine_tidx)
            state["graph"].replay()
            return state["out"]
        except Exception as exc:
            self._policy_graph_disabled = True
            self._policy_graph_state = None
            print(
                f"[server] planner CUDA Graph disabled after capture/replay failure: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return self._policy_forward_normal(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, prompt)

    def _planner_project_cache_enabled(self) -> bool:
        return bool(getattr(self.args, "planner_project_cache", False)) and self.device.type == "cuda"

    def _policy_forward_project_cache(
        self,
        current_coarse: torch.Tensor,
        fine_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tidx: torch.Tensor,
        prompt: list[str],
    ) -> torch.Tensor:
        """Reuse projected coarse history tokens while preserving model I/O."""
        nav = getattr(self.model, "model", self.model)
        planner_amp = self._resolve_amp_dtype(getattr(self.args, "planner_amp", "none"))
        ctx = (
            torch.autocast(device_type="cuda", dtype=planner_amp)
            if self.device.type == "cuda" and planner_amp is not None
            else nullcontext()
        )
        with ctx:
            vis_c_cur = nav.vision_projector(
                current_coarse.unsqueeze(0).to(self.device)
            ).detach()
            self._coarse_hist_proj_tokens.append(vis_c_cur[0])
            hist = list(self._coarse_hist_proj_tokens)
            if len(hist) < self.history:
                first = hist[0]
                hist = [first] * (self.history - len(hist)) + hist
            else:
                hist = hist[-self.history :]
            vis_c = torch.cat(hist, dim=0).unsqueeze(0)
            vis_f = nav.vision_projector(fine_tokens.to(self.device))
            vis_c = nav._insert_temporal_markers(
                vis_c,
                coarse_tidx.to(self.device),
                stream_id=0,
            )
            vis_f = nav._insert_temporal_markers(
                vis_f,
                fine_tidx.to(self.device),
                stream_id=1,
            )
            input_ids, txt_mask = self._tokenize_prompt(prompt)
            txt_emb = nav.backbone.get_input_embeddings()(input_ids)
            control = nav.control_query.expand(vis_c.size(0), 1, -1)
            backbone_dtype = next(nav.backbone.parameters()).dtype
            sequence = torch.cat([txt_emb, vis_c, vis_f, control], dim=1).to(
                backbone_dtype
            )
            attention_mask = torch.cat(
                [
                    txt_mask,
                    torch.ones(
                        vis_c.size(0), vis_c.size(1), dtype=torch.long, device=self.device
                    ),
                    torch.ones(
                        vis_f.size(0), vis_f.size(1), dtype=torch.long, device=self.device
                    ),
                    torch.ones(vis_c.size(0), 1, dtype=torch.long, device=self.device),
                ],
                dim=1,
            )
            out = nav.backbone(
                inputs_embeds=sequence,
                attention_mask=attention_mask,
                use_cache=False,
            )
            control_state = out.last_hidden_state[:, -1].to(
                dtype=next(nav.trajectory_head.parameters()).dtype
            )
            trajectory = nav.trajectory_head(control_state)
            return trajectory * nav.output_scale.to(trajectory.dtype)

    def reset(self) -> None:
        self._coarse_hist_tokens.clear()
        self._coarse_hist_proj_tokens.clear()

    @torch.inference_mode()
    def step(self, rgb_np: np.ndarray, instruction: Optional[str] = None) -> Dict[str, Any]:
        step_t0 = time.perf_counter()

        vision_t0 = time.perf_counter()
        vc, vf, vision_timing = self._encode_frame_tokens(rgb_np)
        self._maybe_cuda_sync()
        vision_encode_ms = (time.perf_counter() - vision_t0) * 1000.0

        self._coarse_hist_tokens.append(vc.detach())

        hist_t0 = time.perf_counter()
        hist_event_start = self._record_cuda_event()
        fine_tokens = vf.unsqueeze(0)
        coarse_tokens = None
        if self._planner_project_cache_enabled():
            coarse_tidx, fine_tidx = self._get_tidx_templates(vc.size(0), fine_tokens.size(1))
        else:
            hist = list(self._coarse_hist_tokens)
            if len(hist) < self.history:
                first = hist[0] if hist else vc
                hist = [first] * (self.history - len(hist)) + hist
            else:
                hist = hist[-self.history :]
            coarse_tokens = torch.cat(hist, dim=0).unsqueeze(0)
            coarse_tidx, fine_tidx = self._get_tidx_templates(hist[0].size(0), fine_tokens.size(1))
        hist_event_end = self._record_cuda_event()
        self._maybe_cuda_sync()
        history_pack_ms = (time.perf_counter() - hist_t0) * 1000.0
        event_timings = []
        if hist_event_start is not None and hist_event_end is not None:
            event_timings.append(("history_pack_ms", hist_event_start, hist_event_end))

        prompt = [instruction or self.default_instruction]
        policy_t0 = time.perf_counter()
        policy_event_start = self._record_cuda_event()
        if self._planner_project_cache_enabled():
            tau = self._policy_forward_project_cache(vc, fine_tokens, coarse_tidx, fine_tidx, prompt)
        else:
            assert coarse_tokens is not None
            tau = self._policy_forward(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, prompt)
        policy_event_end = self._record_cuda_event()
        if policy_event_start is not None and policy_event_end is not None:
            event_timings.append(("policy_forward_ms", policy_event_start, policy_event_end))
        if self._timing_sync_enabled():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            policy_forward_ms = (time.perf_counter() - policy_t0) * 1000.0
            tau_cpu = tau.detach().float().cpu()[0]
        else:
            # The CPU copy is the single required sync point in the fast live path.
            tau_cpu = tau.detach().float().cpu()[0]
            policy_forward_ms = (time.perf_counter() - policy_t0) * 1000.0
        if self._timing_event_enabled():
            self._resolve_cuda_events(vision_timing)
            local_timing: Dict[str, Any] = {"_cuda_events": event_timings}
            self._resolve_cuda_events(local_timing)
            history_pack_ms = float(local_timing.get("history_pack_ms", history_pack_ms))
            policy_forward_ms = float(local_timing.get("policy_forward_ms", policy_forward_ms))
            vision_encode_ms = float(
                vision_timing.get("vision_preprocess_ms", 0.0)
                + vision_timing.get("vision_dino_ms", 0.0)
                + vision_timing.get("vision_siglip_ms", 0.0)
                + vision_timing.get("vision_pool_ms", 0.0)
            )
        tau_np = tau_cpu.numpy().astype(np.float32)

        # Waypoint -> velocity via the shared on-device controller.
        cmd, ctrl_info = self.waypoint_controller.compute(tau_np)
        vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
        e2e_ms = (time.perf_counter() - step_t0) * 1000.0

        return {
            "trajectory": tau_np.tolist(),
            "waypoints": tau_np.tolist(),
            "base_velocity": [vx, vy, wz],
            "controller": ctrl_info,
            "control_dt": float(self.control_dt),
            "instruction": prompt[0],
            "e2e_ms": e2e_ms,
            "vision_encode_ms": vision_encode_ms,
            "vision_preprocess_ms": float(vision_timing.get("vision_preprocess_ms", 0.0)),
            "vision_dino_ms": float(vision_timing.get("vision_dino_ms", 0.0)),
            "vision_siglip_ms": float(vision_timing.get("vision_siglip_ms", 0.0)),
            "vision_pool_ms": float(vision_timing.get("vision_pool_ms", 0.0)),
            "history_pack_ms": history_pack_ms,
            "policy_forward_ms": policy_forward_ms,
        }


app = Flask(__name__)
agent: Optional[MiniCPMRobotTrackRealworldAgent] = None
server_args: Optional[argparse.Namespace] = None
step_idx = 0
eval_lock = threading.Lock()
TCP_REQ_HEADER = struct.Struct("!4sII")
TCP_RESP_HEADER = struct.Struct("!4sI")
TCP_REQ_MAGIC = b"OVL1"
TCP_RESP_MAGIC = b"OVR1"
MAX_TCP_JSON_BYTES = 1 << 20
MAX_TCP_IMAGE_BYTES = 16 << 20
overlay_lock = threading.Lock()
latest_overlay_jpeg: Optional[bytes] = None
latest_model_input_jpeg: Optional[bytes] = None
latest_camera_jpeg: Optional[bytes] = None
latest_camera_frame_seq = 0
latest_camera_update_wall = 0.0
latest_overlay_step = 0
latest_overlay_infer = 0.0
latest_overlay_request = 0.0
latest_overlay_noninfer = 0.0
latest_overlay_velocity = [0.0, 0.0, 0.0]
latest_client_exec_velocity = [0.0, 0.0, 0.0]
latest_client_control_mode = "unknown"
latest_latency_note = "Client timestamp missing; network latency unavailable."
overlay_cv = threading.Condition()
pending_overlay_job: Optional[Tuple[np.ndarray, Dict[str, Any], int, float]] = None
overlay_worker_started = False


def _parse_client_timestamp(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        ts = float(value)
    except Exception:
        return None
    if not np.isfinite(ts) or ts <= 0.0:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return ts if 946684800.0 <= ts <= 4102444800.0 else None


def _extract_client_send_timestamp(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("client_send_timestamp", "client_send_time", "client_timestamp", "timestamp", "ts"):
        ts = _parse_client_timestamp(payload.get(key))
        if ts is not None:
            return ts
    return None


def _estimate_latency_metrics(
    client_send_ts: Optional[float],
    server_recv_wall: float,
    server_resp_wall: float,
    request_time: float,
    infer_time: float,
) -> Dict[str, Any]:
    server_overhead = max(0.0, float(request_time) - float(infer_time))
    metrics: Dict[str, Any] = {
        "server_request_latency": float(request_time),
        "server_inference_latency": float(infer_time),
        "server_overhead_latency": server_overhead,
        "network_upload_latency": None,
        "overall_latency": None,
        "client_send_timestamp": client_send_ts,
        "latency_note": "Client timestamp missing; network latency unavailable.",
    }
    if client_send_ts is None:
        return metrics

    upload = server_recv_wall - client_send_ts
    overall = server_resp_wall - client_send_ts
    if upload < 0.0 or overall < 0.0:
        metrics["latency_note"] = "Client/server clocks appear unsynchronized; network estimate hidden."
        return metrics

    metrics["network_upload_latency"] = float(upload)
    metrics["overall_latency"] = float(overall)
    metrics["latency_note"] = "Overall latency ~= upload network + server request latency."
    return metrics


def _draw_velocity_overlay(
    frame_bgr: np.ndarray,
    model_output: Dict[str, Any],
    step: int,
    infer_time: float,
    max_vx: float,
    max_vy: float,
    max_wz: float,
) -> np.ndarray:
    """Same visualization logic as offline_video_velocity_overlay.py."""
    base = frame_bgr.copy()
    h, w = base.shape[:2]
    panel_h = 130
    vis = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    vis[:h, :, :] = base
    cv2.rectangle(vis, (0, h), (w - 1, h + panel_h - 1), (20, 20, 20), thickness=-1)
    cv2.line(vis, (0, h), (w - 1, h), (90, 90, 90), 1)

    vel = model_output.get("base_velocity", [0.0, 0.0, 0.0])
    if not isinstance(vel, list) or len(vel) < 3:
        vel = [0.0, 0.0, 0.0]
    vx, vy, wz = float(vel[0]), float(vel[1]), float(vel[2])

    waypoints = model_output.get("waypoints", [])
    if not isinstance(waypoints, list):
        waypoints = []

    panel_w = min(520, max(360, w - 20))
    panel_x0 = (w - panel_w) // 2
    panel_x1 = panel_x0 + panel_w
    panel_y0, panel_y1 = h + 12, h + panel_h - 10
    cv2.rectangle(vis, (panel_x0, panel_y0), (panel_x1, panel_y1), (25, 25, 25), thickness=-1)
    cv2.rectangle(vis, (panel_x0, panel_y0), (panel_x1, panel_y1), (220, 220, 220), thickness=1)

    cx, cy = (panel_x0 + panel_x1) // 2, panel_y0 + 58
    sx = 60.0 / max(1e-6, max_vy)
    sy = 60.0 / max(1e-6, max_vx)
    dx = int(np.clip(-vy * sx, -70, 70))
    dy = int(np.clip(-vx * sy, -70, 70))
    ex, ey = cx + dx, cy + dy
    cv2.arrowedLine(vis, (cx, cy), (ex, ey), (30, 80, 255), 3, tipLength=0.25)
    cv2.circle(vis, (cx, cy), 2, (230, 230, 230), -1)

    traj_xy = []
    for wp in waypoints:
        if not isinstance(wp, list) or len(wp) < 2:
            continue
        try:
            wx = float(wp[0])
            wy = float(wp[1])
        except Exception:
            continue
        traj_xy.append((wx, wy))
    if traj_xy:
        max_abs_x = max(abs(p[0]) for p in traj_xy)
        max_abs_y = max(abs(p[1]) for p in traj_xy)
        traj_scale = 58.0 / max(0.20, max_abs_x, max_abs_y)
        pts = []
        for wx, wy in traj_xy:
            px = int(np.clip(cx - wy * traj_scale, panel_x0 + 8, panel_x1 - 8))
            py = int(np.clip(cy - wx * traj_scale, panel_y0 + 10, panel_y1 - 10))
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 200, 255), thickness=2)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 3 if i == 0 else 2, (0, 255, 255), -1)

    turn = "left" if wz > 0.05 else ("right" if wz < -0.05 else "straight")
    cv2.putText(
        vis,
        f"step={step} infer={infer_time:.2f}s",
        (panel_x0 + 10, panel_y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}",
        (panel_x0 + 10, panel_y0 + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"turn={turn} (|wz|max={max_wz:.2f})",
        (panel_x0 + 10, panel_y0 + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Linear dir: up=+vx, left=+vy | Yellow: predicted waypoints",
        (panel_x0 + 10, panel_y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return vis


def _render_overlay_frame(frame_rgb: np.ndarray, model_output: Dict[str, Any], step: int, infer_time: float) -> None:
    global latest_overlay_jpeg, latest_model_input_jpeg, latest_overlay_step, latest_overlay_infer
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    model_ok, model_enc = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    vis_bgr = _draw_velocity_overlay(
        frame_bgr,
        model_output,
        step=step,
        infer_time=infer_time,
        max_vx=float(server_args.max_vx if server_args is not None else 1.2),
        max_vy=float(server_args.max_vy if server_args is not None else 0.8),
        max_wz=float(server_args.max_wz if server_args is not None else 1.5),
    )
    ok, enc = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    with overlay_lock:
        latest_overlay_jpeg = enc.tobytes()
        if model_ok:
            latest_model_input_jpeg = model_enc.tobytes()
        latest_overlay_step = int(step)
        latest_overlay_infer = float(infer_time)


def _overlay_worker_loop() -> None:
    global pending_overlay_job
    while True:
        with overlay_cv:
            while pending_overlay_job is None:
                overlay_cv.wait()
            job = pending_overlay_job
            pending_overlay_job = None
        try:
            frame_rgb, model_output, step, infer_time = job
            _render_overlay_frame(frame_rgb, model_output, step, infer_time)
        except Exception as exc:
            print(f"[overlay] render failed: {type(exc).__name__}: {exc}", flush=True)


def _start_overlay_worker_once() -> None:
    global overlay_worker_started
    with overlay_cv:
        if overlay_worker_started:
            return
        overlay_worker_started = True
    threading.Thread(target=_overlay_worker_loop, daemon=True).start()


def _update_overlay_frame(frame_rgb: np.ndarray, model_output: Dict[str, Any], step: int, infer_time: float) -> None:
    global pending_overlay_job
    mode = str(getattr(server_args, "overlay_mode", "async") if server_args is not None else "async")
    mode = mode.strip().lower()
    if mode == "off":
        return
    if mode == "inline":
        _render_overlay_frame(frame_rgb, model_output, step, infer_time)
        return

    _start_overlay_worker_once()
    frame_copy = np.ascontiguousarray(frame_rgb)
    output_copy = dict(model_output)
    with overlay_cv:
        pending_overlay_job = (frame_copy, output_copy, int(step), float(infer_time))
        overlay_cv.notify()


def _frame_stream(frame_kind: str) -> Iterator[bytes]:
    boundary = b"--frame"
    last_payload = None
    while True:
        with overlay_lock:
            if frame_kind == "camera":
                payload = latest_camera_jpeg
            elif frame_kind == "model-input":
                payload = latest_model_input_jpeg
            else:
                payload = latest_overlay_jpeg
        if payload is None:
            # First frame not arrived yet; keep stream alive.
            time.sleep(0.05)
            continue
        if payload == last_payload:
            time.sleep(0.02)
            continue
        last_payload = payload
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + b"Cache-Control: no-cache\r\n"
            + b"Pragma: no-cache\r\n\r\n"
            + payload
            + b"\r\n"
        )


@app.route("/health", methods=["GET"])
def health() -> Any:
    args = globals().get("server_args", None)
    payload = {"ok": True, "model_loaded": agent is not None}
    if args is not None:
        payload.update(
            {
                "model_dir": str(getattr(args, "model_dir", "")),
                "minicpm_robot_track_root": PROJECT_ROOT,
                "tcp_port": int(getattr(args, "tcp_port", 0) or 0),
            }
        )
    return jsonify(payload)


@app.route("/api/camera-frame", methods=["POST"])
def camera_frame() -> Any:
    global latest_camera_jpeg, latest_camera_frame_seq, latest_camera_update_wall
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify({"error": "camera frame publishing is local-only"}), 403
    jpeg = request.get_data(cache=False)
    if not jpeg or len(jpeg) > (4 << 20) or not jpeg.startswith(b"\xff\xd8"):
        return jsonify({"error": "invalid JPEG frame"}), 400
    try:
        frame_seq = int(request.headers.get("X-Frame-Seq", "0"))
    except ValueError:
        frame_seq = 0
    with overlay_lock:
        latest_camera_jpeg = bytes(jpeg)
        latest_camera_frame_seq = frame_seq
        latest_camera_update_wall = time.time()
    return ("", 204)


@app.route("/web", methods=["GET"])
def web_view() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniCPM-RobotTrack Monitor</title>
  <style>
    * { box-sizing: border-box; }
    body { background:#101214; color:#ddd; font-family: monospace; margin:16px; }
    .wrap { max-width:1400px; margin:0 auto; }
    h2 { margin:0 0 12px; font-size:20px; font-weight:600; }
    .media-grid { display:grid; grid-template-columns:minmax(0, 2fr) minmax(320px, 1fr); gap:12px; align-items:start; }
    .media-title { color:#aab4bf; font-size:13px; margin:0 0 6px; }
    .stream { display:block; width:100%; border:1px solid #333; border-radius:6px; background:#050607; object-fit:contain; }
    .camera-stream { aspect-ratio:16 / 9; }
    .model-stream { aspect-ratio:384 / 514; }
    .grid { display:grid; grid-template-columns:repeat(3, minmax(220px, 1fr)); gap:10px; margin:12px 0 0; }
    .box { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:8px 10px; }
    .k { color:#8ab4f8; font-size:12px; margin-bottom:4px; }
    .v { color:#e6edf3; font-size:14px; }
    @media (max-width:900px) {
      .media-grid, .grid { grid-template-columns:1fr; }
      .model-pane { max-width:520px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>MiniCPM-RobotTrack Realtime Monitor</h2>
    <div class="media-grid">
      <section>
        <div class="media-title">Raw camera</div>
        <img class="stream camera-stream" src="/stream/camera" alt="Raw camera stream" />
      </section>
      <section class="model-pane">
        <div class="media-title">Model input 384×384 · velocity and waypoint</div>
        <img class="stream model-stream" src="/stream/overlay" alt="Model input with control overlay" />
      </section>
    </div>
    <div class="grid">
      <div class="box"><div class="k">Step / Model Infer</div><div class="v" id="infer">-</div></div>
      <div class="box"><div class="k">Server Request Total</div><div class="v" id="req">-</div></div>
      <div class="box"><div class="k">Estimated Non-Infer Delay</div><div class="v" id="noninfer">-</div></div>
      <div class="box"><div class="k">Model Base Velocity (vx, vy, wz)</div><div class="v" id="vel">-</div></div>
      <div class="box"><div class="k">Executed Velocity (client -> dog)</div><div class="v" id="execvel">-</div></div>
      <div class="box"><div class="k">Client Control Mode</div><div class="v" id="ctrlmode">-</div></div>
      <div class="box"><div class="k">Camera Frame / Age</div><div class="v" id="camera">-</div></div>
    </div>
  </div>
  <script>
    function setText(id, txt) { document.getElementById(id).textContent = txt; }
    async function refresh() {
      try {
        const resp = await fetch('/api/state', { cache: 'no-store' });
        const s = await resp.json();
        const infer = Number(s.infer_time || 0).toFixed(3);
        const req = Number(s.request_time || 0).toFixed(3);
        const non = Number(s.noninfer_delay || 0).toFixed(3);
        const v = s.base_velocity || [0,0,0];
        const ev = s.client_exec_velocity || [0,0,0];
        setText('infer', `${s.step} / ${infer}s`);
        setText('req', `${req}s`);
        setText('noninfer', `${non}s`);
        setText('vel', `[${Number(v[0]).toFixed(3)}, ${Number(v[1]).toFixed(3)}, ${Number(v[2]).toFixed(3)}]`);
        setText('execvel', `[${Number(ev[0]).toFixed(3)}, ${Number(ev[1]).toFixed(3)}, ${Number(ev[2]).toFixed(3)}]`);
        setText('ctrlmode', String(s.client_control_mode || 'unknown'));
        setText('camera', `${Number(s.camera_frame_seq || 0)} / ${Number(s.camera_frame_age_ms || 0).toFixed(0)} ms`);
      } catch (_e) {}
    }
    refresh();
    setInterval(refresh, 200);
  </script>
</body>
</html>
"""


@app.route("/stream/overlay", methods=["GET"])
def stream_overlay() -> Response:
    return Response(
        stream_with_context(_frame_stream("overlay")),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/camera", methods=["GET"])
def stream_camera() -> Response:
    return Response(
        stream_with_context(_frame_stream("camera")),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/model-input", methods=["GET"])
def stream_model_input() -> Response:
    return Response(
        stream_with_context(_frame_stream("model-input")),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state", methods=["GET"])
def api_state() -> Any:
    with overlay_lock:
        has_frame = latest_overlay_jpeg is not None
        has_model_input = latest_model_input_jpeg is not None
        has_camera = latest_camera_jpeg is not None
        camera_frame_seq = latest_camera_frame_seq
        camera_update_wall = latest_camera_update_wall
        step = latest_overlay_step
        infer = latest_overlay_infer
        req = latest_overlay_request
        noninfer = latest_overlay_noninfer
        vel = list(latest_overlay_velocity)
        exec_vel = list(latest_client_exec_velocity)
        mode = str(latest_client_control_mode)
        note = str(latest_latency_note)
    return jsonify(
        {
            "ok": True,
            "has_overlay_frame": has_frame,
            "has_model_input_frame": has_model_input,
            "has_camera_frame": has_camera,
            "camera_frame_seq": camera_frame_seq,
            "camera_frame_age_ms": max(0.0, (time.time() - camera_update_wall) * 1000.0) if camera_update_wall > 0 else 0.0,
            "step": step,
            "infer_time": infer,
            "request_time": req,
            "noninfer_delay": noninfer,
            "base_velocity": vel,
            "client_exec_velocity": exec_vel,
            "client_control_mode": mode,
            "latency_note": note,
        }
    )


def _run_eval_predecoded(
    payload: Dict[str, Any],
    image_np: np.ndarray,
    transport: str,
    req_t0: float,
    req_perf_t0: float,
    req_recv_wall: float,
    server_decode_ms: float,
    server_payload_parse_ms: float,
) -> Dict[str, Any]:
    global step_idx
    if agent is None or server_args is None:
        raise RuntimeError("server not initialized")

    with eval_lock:
        if bool(payload.get("reset", False)):
            step_idx = 0
            agent.reset()

        step_idx += 1
        current_step = step_idx
        instruction = str(payload.get("instruction", server_args.instruction))

        infer_t0 = time.time()
        out = agent.step(image_np, instruction)
        infer_time = time.time() - infer_t0

    _update_overlay_frame(image_np, out, current_step, infer_time)

    response_t0 = time.perf_counter()
    out["step"] = current_step
    out["infer_time"] = infer_time
    out["request_time"] = time.time() - req_t0
    noninfer = max(0.0, float(out["request_time"]) - float(out["infer_time"]))
    out["noninfer_delay"] = noninfer
    out["server_decode_ms"] = server_decode_ms
    out["server_payload_parse_ms"] = server_payload_parse_ms
    out["server_pre_infer_ms"] = max(0.0, (infer_t0 - req_t0) * 1000.0)
    out["transport"] = transport
    resp_ready_wall = time.time()
    latency_metrics = _estimate_latency_metrics(
        client_send_ts=_extract_client_send_timestamp(payload),
        server_recv_wall=req_recv_wall,
        server_resp_wall=resp_ready_wall,
        request_time=float(out["request_time"]),
        infer_time=float(out["infer_time"]),
    )
    out.update(latency_metrics)
    out["server_receive_timestamp"] = req_recv_wall
    out["server_response_ready_timestamp"] = resp_ready_wall
    out["server_route_total_ms"] = (time.perf_counter() - req_perf_t0) * 1000.0
    client_exec = payload.get("client_exec_velocity", [0.0, 0.0, 0.0])
    client_mode = str(payload.get("client_control_mode", "unknown"))
    if not isinstance(client_exec, list) or len(client_exec) < 3:
        client_exec = [0.0, 0.0, 0.0]
    out["client_exec_velocity"] = [float(client_exec[0]), float(client_exec[1]), float(client_exec[2])]
    out["client_control_mode"] = client_mode
    for key in (
        "client_serialize_ms",
        "client_camera_header_to_callback_ms",
        "client_camera_callback_ms",
        "client_rgb_bridge_ms",
        "client_rgb_serialize_ms",
        "client_depth_bridge_ms",
        "client_depth_serialize_ms",
        "client_callback_to_http_send_ms",
    ):
        if key in payload:
            try:
                out[key] = float(payload[key])
            except Exception:
                pass
    out["server_response_prepare_ms"] = (time.perf_counter() - response_t0) * 1000.0
    with overlay_lock:
        global latest_overlay_request, latest_overlay_noninfer, latest_overlay_velocity
        global latest_client_exec_velocity, latest_client_control_mode, latest_latency_note
        latest_overlay_request = float(out["request_time"])
        latest_overlay_noninfer = float(noninfer)
        vel = out.get("base_velocity", [0.0, 0.0, 0.0])
        if isinstance(vel, list) and len(vel) >= 3:
            latest_overlay_velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        else:
            latest_overlay_velocity = [0.0, 0.0, 0.0]
        latest_client_exec_velocity = [float(client_exec[0]), float(client_exec[1]), float(client_exec[2])]
        latest_client_control_mode = client_mode
        latest_latency_note = str(out.get("latency_note", latest_latency_note))
    return _format_eval_response(out, str(getattr(server_args, "response_mode", "full")))


def _recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    got = 0
    while got < nbytes:
        n = conn.recv_into(view[got:], nbytes - got)
        if n == 0:
            raise ConnectionError("peer closed")
        got += n
    return bytes(buf)


def _tcp_client_loop(conn: socket.socket, addr: Any) -> None:
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    print(f"[tcp] client connected: {addr}", flush=True)
    try:
        while True:
            magic, json_len, image_len = TCP_REQ_HEADER.unpack(_recv_exact(conn, TCP_REQ_HEADER.size))
            req_t0 = time.time()
            req_perf_t0 = time.perf_counter()
            req_recv_wall = req_t0
            if magic != TCP_REQ_MAGIC:
                raise ValueError(f"bad TCP request magic: {magic!r}")
            if json_len <= 0 or json_len > MAX_TCP_JSON_BYTES:
                raise ValueError(f"bad TCP json length: {json_len}")
            if image_len <= 0 or image_len > MAX_TCP_IMAGE_BYTES:
                raise ValueError(f"bad TCP image length: {image_len}")

            payload_t0 = time.perf_counter()
            payload = json.loads(_recv_exact(conn, json_len).decode("utf-8"))
            server_payload_parse_ms = (time.perf_counter() - payload_t0) * 1000.0

            decode_t0 = time.perf_counter()
            image_encoding = str(payload.get("image_encoding", "jpeg")).lower()
            image_payload = _recv_exact(conn, image_len)
            if image_encoding in ("raw_rgb", "rgb"):
                width = int(payload.get("image_width", 0) or 0)
                height = int(payload.get("image_height", 0) or 0)
                channels = int(payload.get("image_channels", 3) or 3)
                expected = width * height * channels
                if width <= 0 or height <= 0 or channels != 3 or expected != image_len:
                    raise ValueError(
                        f"bad raw_rgb shape width={width} height={height} channels={channels} bytes={image_len}"
                    )
                image_np = np.frombuffer(image_payload, dtype=np.uint8).reshape((height, width, channels))
                transport = "tcp_raw"
            else:
                image = Image.open(io.BytesIO(image_payload)).convert("RGB")
                image_np = np.asarray(image)
                transport = "tcp_jpeg"
            server_decode_ms = (time.perf_counter() - decode_t0) * 1000.0

            result = _run_eval_predecoded(
                payload=payload,
                image_np=image_np,
                transport=transport,
                req_t0=req_t0,
                req_perf_t0=req_perf_t0,
                req_recv_wall=req_recv_wall,
                server_decode_ms=server_decode_ms,
                server_payload_parse_ms=server_payload_parse_ms,
            )
            resp = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            conn.sendall(TCP_RESP_HEADER.pack(TCP_RESP_MAGIC, len(resp)) + resp)
    except ConnectionError:
        pass
    except Exception as exc:
        try:
            err = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, separators=(",", ":")).encode("utf-8")
            conn.sendall(TCP_RESP_HEADER.pack(TCP_RESP_MAGIC, len(err)) + err)
        except Exception:
            pass
        print(f"[tcp] client error {addr}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[tcp] client disconnected: {addr}", flush=True)


def _tcp_server_loop(host: str, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f"[tcp] inference server listening on {host}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_tcp_client_loop, args=(conn, addr), daemon=True).start()


def _start_tcp_server_if_requested() -> None:
    if server_args is None:
        return
    port = int(getattr(server_args, "tcp_port", 0) or 0)
    if port <= 0:
        return
    host = str(getattr(server_args, "tcp_host", "") or getattr(server_args, "host", "0.0.0.0"))
    threading.Thread(target=_tcp_server_loop, args=(host, port), daemon=True).start()


@app.route("/eval_dual", methods=["POST"])
def eval_dual() -> Any:
    global step_idx
    if agent is None or server_args is None:
        return jsonify({"error": "server not initialized"}), 500

    req_t0 = time.time()
    req_perf_t0 = time.perf_counter()
    req_recv_wall = req_t0

    payload_t0 = time.perf_counter()
    payload: Dict[str, Any] = {}
    if "json" in request.form:
        try:
            payload = json.loads(request.form["json"])
        except Exception:
            payload = {}
    elif request.is_json:
        payload = request.get_json(silent=True) or {}
    server_payload_parse_ms = (time.perf_counter() - payload_t0) * 1000.0
    client_send_ts = _extract_client_send_timestamp(payload)

    decode_t0 = time.perf_counter()
    if str(payload.get("transport", "")).lower() == "shm" or "image_shm_name" in payload:
        shm_name = str(payload.get("image_shm_name", ""))
        h = int(payload.get("image_height", 0))
        w = int(payload.get("image_width", 0))
        c = int(payload.get("image_channels", 3))
        if not shm_name or h <= 0 or w <= 0 or c != 3:
            return jsonify({"error": "shm image requires image_shm_name/image_height/image_width/image_channels=3"}), 400
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            image_np = np.ndarray((h, w, c), dtype=np.uint8, buffer=shm.buf).copy()
        finally:
            shm.close()
        transport = "shm_rgb"
    elif "image_raw" in request.files:
        raw_file = request.files["image_raw"]
        h = int(payload.get("image_height", 0))
        w = int(payload.get("image_width", 0))
        c = int(payload.get("image_channels", 3))
        if h <= 0 or w <= 0 or c != 3:
            return jsonify({"error": "raw image requires image_height/image_width/image_channels=3"}), 400
        raw = raw_file.stream.read()
        expected = h * w * c
        if len(raw) != expected:
            return jsonify({"error": f"raw image byte size mismatch: {len(raw)} != {expected}"}), 400
        image_np = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c).copy()
        transport = "raw_rgb"
    elif "image" in request.files:
        img_file = request.files["image"]
        image = Image.open(img_file.stream).convert("RGB")
        image_np = np.asarray(image)
        transport = "jpeg"
    else:
        return jsonify({"error": "missing form file field: image or image_raw"}), 400
    server_decode_ms = (time.perf_counter() - decode_t0) * 1000.0

    # Keep payload compatibility (field exists but currently unused here).
    if "depth" in request.files:
        _ = request.files["depth"]

    return jsonify(
        _run_eval_predecoded(
            payload=payload,
            image_np=image_np,
            transport=transport,
            req_t0=req_t0,
            req_perf_t0=req_perf_t0,
            req_recv_wall=req_recv_wall,
            server_decode_ms=server_decode_ms,
            server_payload_parse_ms=server_payload_parse_ms,
        )
    )


def _format_eval_response(out: Dict[str, Any], response_mode: str) -> Dict[str, Any]:
    """Optionally strip fields from the eval response for the selected client mode.

    Modes:
        - full        : return everything (debug, full timing, trajectory, velocity, controller info)
        - trajectory  : drop velocity-only fields
        - velocity    : drop trajectory/waypoints; keep base_velocity + metrics
        - control     : keep waypoints + velocity, but drop the duplicate trajectory field
    """
    mode = (response_mode or "full").strip().lower()
    if mode == "full":
        return out
    trimmed = dict(out)
    if mode == "velocity":
        for k in ("trajectory", "waypoints", "decode_waypoint", "decode_waypoint_idx"):
            trimmed.pop(k, None)
    elif mode == "control":
        for k in ("trajectory", "decode_waypoint", "decode_waypoint_idx"):
            trimmed.pop(k, None)
    elif mode == "trajectory":
        # keep both but signal intent
        pass
    return trimmed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--control_dt", type=float, default=0.1)

    parser.add_argument("--max_vx", type=float, default=1.2)
    parser.add_argument("--max_vy", type=float, default=0.8)
    parser.add_argument("--max_wz", type=float, default=1.5)

    parser.add_argument(
        "--instruction",
        type=str,
        default="Follow the target person without collision.",
    )
    parser.add_argument(
        "--vision_amp",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
        help="Vision tower autocast dtype. bf16 is the validated fast/stable path.",
    )
    parser.add_argument(
        "--planner_amp",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
        help="Planner forward autocast dtype.",
    )
    parser.add_argument(
        "--dino_backend",
        type=str,
        default="torch",
        choices=["torch", "trt_direct"],
        help="DINO backend. 'trt_direct' runs exported DINO patch-token TensorRT in this process.",
    )
    parser.add_argument(
        "--dino_trt_engine",
        type=str,
        default=DEFAULT_DINO_ENGINE,
        help="Path to the DINO patch-token TRT engine.",
    )
    parser.add_argument(
        "--dino_trt_output_tokens",
        type=int,
        default=576,
        help="DINO TRT output token count. 384x384 with patch16 uses 576.",
    )
    parser.add_argument(
        "--siglip_backend",
        type=str,
        default="torch",
        choices=["torch", "trt_direct"],
        help="SigLIP backend. 'trt_direct' runs TensorRT in this process.",
    )
    parser.add_argument(
        "--siglip_trt_engine",
        type=str,
        default=DEFAULT_SIGLIP_ENGINE,
        help="Path to the SigLIP TRT engine.",
    )
    parser.add_argument(
        "--siglip_trt_output_tokens",
        type=int,
        default=729,
        help="SigLIP TRT output token count. Default 729; pooled 24x24 engines use 576.",
    )
    parser.add_argument(
        "--tcp-host",
        type=str,
        default="",
        help="Host for the optional persistent TCP JPEG inference protocol. Empty means --host.",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=0,
        help="Enable optional persistent TCP JPEG inference protocol on this port. 0 disables.",
    )

    parser.add_argument(
        "--response-mode",
        type=str,
        default="full",
        choices=["full", "trajectory", "velocity", "control"],
        help=(
            "Trim the /eval_dual response payload. 'velocity' drops trajectory/waypoints; "
            "'control' keeps waypoints and velocity but drops the duplicate trajectory."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        type=str,
        default="accurate",
        choices=["accurate", "fast", "event"],
        help="accurate synchronizes per stage; fast minimizes syncs; event reports CUDA event timings with only the final required sync.",
    )
    parser.add_argument(
        "--planner-cuda-graph",
        action="store_true",
        help="Capture fixed-shape MiniCPM planner forward with CUDA Graph. Falls back automatically if capture is unsupported.",
    )
    parser.add_argument(
        "--planner-cuda-graph-warmup",
        type=int,
        default=3,
        help="Warmup forwards before planner CUDA Graph capture.",
    )
    parser.add_argument(
        "--planner-project-cache",
        action="store_true",
        help="Cache projected coarse history tokens and project only the newest coarse frame each step.",
    )
    parser.add_argument(
        "--overlay-mode",
        type=str,
        default="async",
        choices=["async", "inline", "off"],
        help="Web overlay rendering mode. async keeps JPEG drawing off the inference response path.",
    )

    # Waypoint -> velocity controller.
    parser.add_argument(
        "--velocity-controller",
        type=str,
        default="direct",
        choices=["direct", "lookahead"],
        help="Waypoint-to-velocity policy. The validated default is 'direct'.",
    )
    parser.add_argument("--controller-direct-wp-idx", type=int, default=1,
                        help="(direct mode) Which waypoint to target. Divisor is wp_idx*control_dt.")
    parser.add_argument("--controller-lookahead-idx", type=int, default=2,
                        help="(lookahead mode) Feedback target waypoint index.")
    parser.add_argument("--controller-feedforward-points", type=int, default=3,
                        help="(lookahead mode) Number of leading waypoints averaged for feedforward velocity.")
    parser.add_argument("--controller-ff-blend", type=float, default=0.65,
                        help="(lookahead mode) Blend between feedforward (1.0) and feedback (0.0).")
    parser.add_argument("--controller-kx", type=float, default=1.2,
                        help="(lookahead mode) Translational P-gain on target_x.")
    parser.add_argument("--controller-ky", type=float, default=1.0,
                        help="(lookahead mode) Translational P-gain on target_y (only if disable_lateral=False).")
    parser.add_argument("--controller-ktheta", type=float, default=1.4,
                        help="(lookahead mode) Yaw P-gain on target_theta.")
    parser.add_argument("--controller-curvature-gain", type=float, default=0.35,
                        help="(lookahead mode) Curvature feedforward gain.")
    parser.add_argument("--controller-min-turn-speed", type=float, default=0.08,
                        help="(lookahead mode) Floor on base_vx used by curvature term.")
    parser.add_argument("--allow-reverse", action="store_true",
                        help="If set, allow vx<0. Default off (clamp to >=0).")
    parser.add_argument("--disable-lateral", action="store_true", default=True,
                        help="If set, zero out vy regardless of model output. Default ON (quadruped).")
    # ── Turn-in-place coupling: suppress vx when |wz| is large. ──
    parser.add_argument("--turn-in-place-factor", type=float, default=0.0,
                        help="0=off (default). 1=at full yaw, vx drops to min_vx_factor. Try 0.7-1.0 to fix 'rushes forward while turning slow'.")
    parser.add_argument("--min-vx-factor", type=float, default=0.2,
                        help="Floor for vx scale when turn-in-place is active. 0.2=vx caps at 20%% during full yaw.")
    return parser


if __name__ == "__main__":
    server_args = build_parser().parse_args()
    # Perf toggles. Matches the path test_e2e_speed.py used to reach baseline.
    # Note: cudnn.benchmark=True was tested on this Jetson + cudnn 8.6 and reproducibly
    # regressed both DINO and SigLIP (e2e 849 -> 1010 ms). Leaving it OFF.
    agent = MiniCPMRobotTrackRealworldAgent(server_args)
    agent.reset()
    _start_tcp_server_if_requested()
    print(f"[server] start at http://{server_args.host}:{server_args.port}")
    app.run(host=server_args.host, port=server_args.port, threaded=True)
