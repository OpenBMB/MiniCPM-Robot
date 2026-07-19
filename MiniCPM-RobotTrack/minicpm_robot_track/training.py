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

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .checkpoint import save_checkpoint
from .config import ModelConfig
from .data import (
    TrackingDataConfig,
    TrackingDataset,
    collate_tracking_samples,
    infer_data_root,
)
from .modeling import MiniCPMRobotTrack


DEFAULT_HF_CHECKPOINT = Path(
    "minicpm_robot_track/checkpoints/MiniCPM-RobotTrack"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_trajectory_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target trajectories must have identical shapes")
    mask = valid_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * prediction.size(-1)
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return ((prediction - target).square() * mask).sum() / denominator


def cosine_lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _trainable(parameters: Iterable[nn.Parameter]) -> List[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def build_optimizer(model: MiniCPMRobotTrack, args: argparse.Namespace) -> AdamW:
    definitions = (
        ("backbone", model.backbone.parameters(), args.backbone_lr),
        ("vision_projector", model.vision_projector.parameters(), args.projector_lr),
        ("temporal_markers", model.temporal_markers.parameters(), args.temporal_lr),
        ("trajectory_head", model.trajectory_head.parameters(), args.head_lr),
        ("control_query", [model.control_query], args.control_query_lr),
    )
    groups = []
    for name, parameters, configured_lr in definitions:
        values = _trainable(parameters)
        if values:
            base_lr = args.lr if configured_lr is None else configured_lr
            groups.append(
                {
                    "params": values,
                    "lr": base_lr,
                    "initial_lr": base_lr,
                    "weight_decay": args.weight_decay,
                    "name": name,
                }
            )
    if not groups:
        raise ValueError("the model has no trainable parameters")
    return AdamW(groups, betas=(0.9, 0.95))


def _task_sources(args: argparse.Namespace) -> List[Tuple[str, Path, Path]]:
    sources: List[Tuple[str, Path, Path]] = []
    for task in ("stt", "at", "dt"):
        json_source: Optional[Path] = getattr(args, f"{task}_json")
        cache_root: Optional[Path] = getattr(args, f"{task}_cache")
        if json_source is None:
            if cache_root is not None:
                raise ValueError(f"--{task}-cache requires --{task}-json")
            continue
        if cache_root is None:
            cache_root = infer_data_root(json_source) / "vision_cache"
        sources.append((task, json_source, cache_root))
    if not sources:
        raise ValueError("provide at least one of --stt-json, --at-json, or --dt-json")
    return sources


def build_dataset(args: argparse.Namespace) -> Dataset:
    datasets: List[Dataset] = []
    for task, json_source, cache_root in _task_sources(args):
        datasets.append(
            TrackingDataset(
                TrackingDataConfig(
                    json_source=json_source,
                    cache_root=cache_root,
                    task_name=task,
                    num_waypoints=args.num_waypoints,
                    history_frames=args.history,
                )
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _checkpoint_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    fields = (
        "init_checkpoint",
        "train_from_base",
        "backbone",
        "history",
        "num_waypoints",
        "dropout",
        "xy_scale",
        "no_tanh_actions",
        "freeze_backbone",
        "gradient_checkpointing",
        "epochs",
        "batch_size",
        "lr",
        "backbone_lr",
        "projector_lr",
        "temporal_lr",
        "head_lr",
        "control_query_lr",
        "weight_decay",
        "warmup_ratio",
        "min_lr_ratio",
        "grad_clip",
        "precision",
        "seed",
    )
    values = {field: getattr(args, field) for field in fields}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def _model_config(args: argparse.Namespace, vision_feature_dim: int) -> ModelConfig:
    return ModelConfig(
        backbone_name=args.backbone,
        vision_feature_dim=vision_feature_dim,
        history_frames=args.history,
        num_waypoints=args.num_waypoints,
        trajectory_dropout=args.dropout,
        xy_scale=args.xy_scale,
        use_tanh_actions=not args.no_tanh_actions,
        freeze_backbone=args.freeze_backbone,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def _load_hf_initial_model(
    checkpoint: Path,
    config: ModelConfig,
) -> MiniCPMRobotTrack:
    """Build the trainable package model from a released local HF snapshot."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    released_model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    backbone = getattr(released_model, "backbone", None)
    if backbone is None:
        raise ValueError(
            f"HF checkpoint does not expose the MiniCPM backbone: {checkpoint}"
        )

    released_config = released_model.config
    released_backbone_name = getattr(released_config, "backbone_name", None)
    if released_backbone_name:
        config.backbone_name = str(released_backbone_name)
        config.validate()

    expected_values = {
        "vision_feature_dim": config.vision_feature_dim,
        "coarse_tokens_per_frame": config.coarse_tokens_per_frame,
        "fine_tokens_current_frame": config.fine_tokens_current_frame,
        "num_waypoints": config.num_waypoints,
        "action_dim": config.action_dim,
        "max_time_steps": config.max_time_steps,
    }
    mismatched = {
        name: (getattr(released_config, name, None), expected)
        for name, expected in expected_values.items()
        if getattr(released_config, name, None) != expected
    }
    if mismatched:
        details = ", ".join(
            f"{name}=checkpoint:{actual!r}/training:{expected!r}"
            for name, (actual, expected) in sorted(mismatched.items())
        )
        raise ValueError(f"HF checkpoint architecture does not match training: {details}")

    state = released_model.state_dict()
    model = MiniCPMRobotTrack(
        config,
        backbone=backbone,
        tokenizer=tokenizer,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"HF checkpoint weights do not match the training model: {checkpoint}"
        ) from error

    # output_scale is stored in the snapshot, but a deliberate --xy-scale
    # override should still take effect for the new training run.
    with torch.no_grad():
        model.output_scale.fill_(1.0)
        model.output_scale[..., :2] = config.xy_scale
    return model


def build_model(
    args: argparse.Namespace,
    vision_feature_dim: int,
) -> MiniCPMRobotTrack:
    config = _model_config(args, vision_feature_dim)
    if args.init_checkpoint is None:
        return MiniCPMRobotTrack(config)
    return _load_hf_initial_model(args.init_checkpoint, config)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = build_dataset(args)
    first_sample = dataset[0]
    vision_feature_dim = int(first_sample["fine_tokens"].size(-1))

    model = build_model(args, vision_feature_dim).to(device)
    if args.precision == "fp32":
        model.float()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_tracking_samples,
    )
    if len(loader) == 0:
        raise ValueError("training dataset is empty")

    optimizer = build_optimizer(model, args)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    use_bfloat16 = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_index, batch in enumerate(loader):
            lr_scale = cosine_lr_scale(
                global_step, total_steps, warmup_steps, args.min_lr_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_scale

            tensor_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bfloat16
                else nullcontext()
            )
            with autocast:
                predicted = model(
                    tensor_batch["coarse_tokens"],
                    tensor_batch["coarse_time_indices"],
                    tensor_batch["fine_tokens"],
                    tensor_batch["fine_time_indices"],
                    batch["instruction"],
                )
                predicted_normalized = model.normalize_trajectory(predicted)
                target_normalized = model.normalize_trajectory(tensor_batch["trajectory"])
                loss = masked_trajectory_mse(
                    predicted_normalized,
                    target_normalized,
                    tensor_batch["valid_mask"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            global_step += 1
            if global_step % args.log_every == 0:
                average = running_loss / (batch_index + 1)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={average:.6f} lr_scale={lr_scale:.6f}"
                )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        model,
        epoch=args.epochs,
        global_step=global_step,
        training_config=_checkpoint_training_config(args),
    )
    print(f"saved final checkpoint to {output_dir / 'checkpoint_final.pt'}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MiniCPM-RobotTrack")
    data = parser.add_argument_group("tracking data")
    for task in ("stt", "at", "dt"):
        data.add_argument(f"--{task}-json", type=Path, default=None)
        data.add_argument(f"--{task}-cache", type=Path, default=None)

    model = parser.add_argument_group("model")
    initialization = model.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_HF_CHECKPOINT,
        help="Local released MiniCPM-RobotTrack Hugging Face snapshot",
    )
    initialization.add_argument(
        "--train-from-base",
        action="store_true",
        help="Initialize from --backbone instead of the released tracking snapshot",
    )
    model.add_argument(
        "--backbone",
        default="openbmb/MiniCPM4-0.5B",
        help="MiniCPM backbone used only with --train-from-base",
    )
    model.add_argument("--history", type=int, default=31)
    model.add_argument("--num-waypoints", type=int, default=8)
    model.add_argument("--dropout", type=float, default=0.4)
    model.add_argument("--xy-scale", type=float, default=2.0)
    model.add_argument("--no-tanh-actions", action="store_true")
    model.add_argument("--freeze-backbone", action="store_true")
    model.add_argument("--gradient-checkpointing", action="store_true")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=3)
    optimization.add_argument("--batch-size", type=int, default=8)
    optimization.add_argument("--lr", type=float, default=2e-5)
    optimization.add_argument("--backbone-lr", type=float, default=None)
    optimization.add_argument("--projector-lr", type=float, default=None)
    optimization.add_argument("--temporal-lr", type=float, default=None)
    optimization.add_argument("--head-lr", type=float, default=None)
    optimization.add_argument("--control-query-lr", type=float, default=None)
    optimization.add_argument("--weight-decay", type=float, default=0.01)
    optimization.add_argument("--warmup-ratio", type=float, default=0.03)
    optimization.add_argument("--min-lr-ratio", type=float, default=0.1)
    optimization.add_argument("--grad-clip", type=float, default=1.0)
    optimization.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    if args.train_from_base:
        args.init_checkpoint = None
    elif not args.init_checkpoint.is_dir():
        parser.error(
            "--init-checkpoint must be a complete local Hugging Face snapshot "
            f"directory: {args.init_checkpoint}"
        )

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.history < 0 or args.num_waypoints < 2:
        parser.error("--history must be non-negative and --num-waypoints must be at least 2")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in [0, 1]")
    if args.grad_clip <= 0.0 or args.log_every <= 0:
        parser.error("--grad-clip and --log-every must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    learning_rates = (
        args.lr,
        args.backbone_lr,
        args.projector_lr,
        args.temporal_lr,
        args.head_lr,
        args.control_query_lr,
    )
    if any(value is not None and value <= 0.0 for value in learning_rates):
        parser.error("all configured learning rates must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative")
    if Path(args.backbone).is_absolute():
        parser.error("--backbone must be a Hugging Face model id or a relative local path")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        train(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
