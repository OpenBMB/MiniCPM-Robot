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
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
from PIL import Image

from .data import (
    IndexedExamples,
    cache_path,
    infer_data_root,
    relative_image_path,
)
from .vision import DualVisionEncoder, VisionEncoderConfig


def iter_unique_images(examples: IndexedExamples) -> Iterator[Path]:
    seen = set()
    for example in examples:
        values: Sequence = [*example.get("images", []), example["current"]]
        for value in values:
            image = relative_image_path(value)
            key = image.as_posix()
            if key not in seen:
                seen.add(key)
                yield image


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu().half(), temporary)
    temporary.replace(path)


def _uncached_images(images: Iterator[Path], cache_root: Path) -> Iterator[Path]:
    for image in images:
        if not cache_path(cache_root, image, "coarse").is_file() or not cache_path(
            cache_root, image, "fine"
        ).is_file():
            yield image


def precompute(
    json_source: Path,
    cache_root: Path,
    data_root: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    examples = IndexedExamples(json_source)
    pending = list(_uncached_images(iter_unique_images(examples), cache_root))
    if not pending:
        print("All requested visual features are already cached.")
        return

    encoder = DualVisionEncoder(VisionEncoderConfig(), device=device)
    for start in range(0, len(pending), batch_size):
        relative_paths = pending[start : start + batch_size]
        pil_images: List[Image.Image] = []
        for relative_path in relative_paths:
            absolute_path = data_root / relative_path
            if not absolute_path.is_file():
                raise FileNotFoundError(f"image does not exist: {absolute_path}")
            with Image.open(absolute_path) as image:
                pil_images.append(image.convert("RGB"))

        coarse, fine = encoder.encode_pooled(pil_images)
        for index, relative_path in enumerate(relative_paths):
            _save_tensor(cache_path(cache_root, relative_path, "coarse"), coarse[index])
            _save_tensor(cache_path(cache_root, relative_path, "fine"), fine[index])
        print(f"cached {min(start + batch_size, len(pending))}/{len(pending)} images")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute fixed visual tokens")
    parser.add_argument("--json", type=Path, required=True, help="JSON, JSONL, or JSONL directory")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    data_root = args.data_root or infer_data_root(args.json)
    precompute(
        json_source=args.json,
        cache_root=args.cache_root,
        data_root=data_root,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
