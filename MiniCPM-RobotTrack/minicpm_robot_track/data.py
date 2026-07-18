from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrackingDataConfig:
    json_source: Path
    cache_root: Path
    task_name: str
    num_waypoints: int = 8
    history_frames: int = 31
    coarse_tokens_per_frame: int = 4
    fine_tokens_current_frame: int = 64
    default_dt: float = 0.1
    data_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.task_name.lower() not in {"stt", "at", "dt"}:
            raise ValueError("task_name must be stt, at, or dt")
        if self.num_waypoints < 2 or self.history_frames < 0:
            raise ValueError("invalid trajectory or history length")
        for name, value in (
            ("coarse_tokens_per_frame", self.coarse_tokens_per_frame),
            ("fine_tokens_current_frame", self.fine_tokens_current_frame),
        ):
            side = int(round(value**0.5)) if value > 0 else 0
            if side * side != value:
                raise ValueError(f"{name} must be a positive square number")


def discover_json_files(source: Path) -> List[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported dataset file: {source}")
        return [source]
    if source.is_dir():
        files = sorted(source.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no JSONL files found under {source}")
        return files
    raise FileNotFoundError(f"dataset source does not exist: {source}")


def infer_data_root(source: Path) -> Path:
    candidate = source.resolve() if source.is_dir() else source.resolve().parent
    for _ in range(6):
        if (candidate / "frames").is_dir():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    raise FileNotFoundError(
        f"could not locate a frames directory above {source}; pass data_root explicitly"
    )


class IndexedExamples:
    """Indexes JSONL byte offsets while keeping large datasets lazy."""

    def __init__(self, source: Path) -> None:
        self.files = discover_json_files(source)
        self._json_examples: Optional[List[Dict[str, Any]]] = None
        self._offsets: List[Tuple[Path, int]] = []

        if len(self.files) == 1 and self.files[0].suffix.lower() == ".json":
            with self.files[0].open("r", encoding="utf-8") as handle:
                values = json.load(handle)
            if not isinstance(values, list) or not values:
                raise ValueError("a JSON dataset must contain a non-empty list")
            self._json_examples = values
            return

        for path in self.files:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        self._offsets.append((path, offset))
        if not self._offsets:
            raise ValueError(f"no examples found in {source}")

    def __len__(self) -> int:
        return len(self._json_examples) if self._json_examples is not None else len(self._offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._json_examples is not None:
            return self._json_examples[index]
        path, offset = self._offsets[index]
        with path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline().decode("utf-8"))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


def load_token_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, dict):
        for key in ("tokens", "features", "V", "Vfine", "Vcoarse"):
            tensor = value.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor.squeeze(0).float() if tensor.ndim == 3 else tensor.float()
    raise ValueError(f"unrecognized visual cache file: {path}")


def cache_path(cache_root: Path, image_path: Path, kind: str) -> Path:
    suffix = {"coarse": "_vcoarse.pt", "fine": "_vfine.pt"}.get(kind)
    if suffix is None:
        raise ValueError(f"unknown cache kind: {kind}")
    return cache_root / image_path.parent / f"{image_path.stem}{suffix}"


def relative_image_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(
            "dataset image paths must be relative to the dataset root; absolute paths are not supported"
        )
    if ".." in path.parts:
        raise ValueError(f"dataset image path cannot escape its root: {path}")
    return path


def integrate_actions_to_waypoints(
    actions: Sequence[Sequence[float]], num_waypoints: int, dt: float
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("actions must be a non-empty [T, D] array")

    vx = values[:, 0]
    vy = values[:, 1] if values.shape[1] > 1 else np.zeros_like(vx)
    yaw_rate = values[:, 2] if values.shape[1] > 2 else np.zeros_like(vx)
    trajectory = np.zeros((values.shape[0], 3), dtype=np.float32)
    for step in range(1, values.shape[0]):
        previous_yaw = trajectory[step - 1, 2]
        trajectory[step, 2] = previous_yaw + yaw_rate[step - 1] * dt
        cosine, sine = np.cos(previous_yaw), np.sin(previous_yaw)
        trajectory[step, 0] = trajectory[step - 1, 0] + (
            cosine * vx[step - 1] - sine * vy[step - 1]
        ) * dt
        trajectory[step, 1] = trajectory[step - 1, 1] + (
            sine * vx[step - 1] + cosine * vy[step - 1]
        ) * dt
    sample_indices = np.linspace(0, len(trajectory) - 1, num_waypoints).round().astype(int)
    return trajectory[sample_indices]


class TrackingDataset(Dataset):
    def __init__(self, config: TrackingDataConfig) -> None:
        self.config = config
        self.examples = IndexedExamples(config.json_source)
        self.data_root = (
            config.data_root.resolve()
            if config.data_root is not None
            else infer_data_root(config.json_source)
        )
        self.cache_root = config.cache_root.resolve()

    def __len__(self) -> int:
        return len(self.examples)

    def _load_cached(self, image: Path, kind: str, expected_tokens: int) -> torch.Tensor:
        path = cache_path(self.cache_root, image, kind)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing visual cache {path}; run minicpm-robot-track-cache first"
            )
        tokens = load_token_tensor(path)
        if tokens.ndim != 2 or tokens.size(0) != expected_tokens:
            raise ValueError(
                f"cache {path} has shape {tuple(tokens.shape)}, expected [{expected_tokens}, C]"
            )
        return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        current = relative_image_path(example["current"])
        fine = self._load_cached(
            current, "fine", self.config.fine_tokens_current_frame
        )

        history_paths = [relative_image_path(value) for value in example.get("images", [])]
        history_paths = history_paths[-self.config.history_frames :]
        history_tokens = [
            self._load_cached(path, "coarse", self.config.coarse_tokens_per_frame)
            for path in history_paths
        ]
        if not history_tokens and self.config.history_frames > 0:
            history_tokens = [
                self._load_cached(
                    current, "coarse", self.config.coarse_tokens_per_frame
                )
            ]
        if history_tokens:
            missing = self.config.history_frames - len(history_tokens)
            history_tokens = [history_tokens[0]] * max(0, missing) + history_tokens
            history_tokens = history_tokens[-self.config.history_frames :]
            coarse = torch.cat(history_tokens, dim=0)
            coarse_times = torch.arange(self.config.history_frames).repeat_interleave(
                self.config.coarse_tokens_per_frame
            )
        else:
            coarse = fine.new_empty((0, fine.size(1)))
            coarse_times = torch.empty(0, dtype=torch.long)

        fine_times = torch.full(
            (fine.size(0),), self.config.history_frames, dtype=torch.long
        )
        if "waypoints" in example:
            trajectory = torch.as_tensor(example["waypoints"], dtype=torch.float32)
        elif "actions" in example:
            dt = float(example.get("dt", self.config.default_dt))
            trajectory = torch.from_numpy(
                integrate_actions_to_waypoints(
                    example["actions"], self.config.num_waypoints, dt
                )
            )
        else:
            raise ValueError("tracking example requires waypoints or actions")

        expected_shape = (self.config.num_waypoints, 3)
        if tuple(trajectory.shape) != expected_shape:
            raise ValueError(
                f"trajectory has shape {tuple(trajectory.shape)}, expected {expected_shape}"
            )
        if "valid_mask" in example:
            valid_mask = torch.as_tensor(example["valid_mask"], dtype=torch.bool)
        elif "valid_idx" in example:
            valid_mask = torch.zeros(self.config.num_waypoints, dtype=torch.bool)
            valid_mask[torch.as_tensor(example["valid_idx"], dtype=torch.long)] = True
        else:
            valid_mask = torch.ones(self.config.num_waypoints, dtype=torch.bool)
        if tuple(valid_mask.shape) != (self.config.num_waypoints,):
            raise ValueError("valid waypoint mask has an unexpected shape")

        return {
            "coarse_tokens": coarse,
            "coarse_time_indices": coarse_times,
            "fine_tokens": fine,
            "fine_time_indices": fine_times,
            "trajectory": trajectory,
            "valid_mask": valid_mask,
            "instruction": str(example.get("instruction", "Follow the target person.")),
            "task": self.config.task_name,
        }


def collate_tracking_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = (
        "coarse_tokens",
        "coarse_time_indices",
        "fine_tokens",
        "fine_time_indices",
        "trajectory",
        "valid_mask",
    )
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    batch["instruction"] = [sample["instruction"] for sample in samples]
    batch["task"] = [sample["task"] for sample in samples]
    return batch
