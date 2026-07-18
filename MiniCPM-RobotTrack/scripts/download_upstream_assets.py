#!/usr/bin/env python3
"""Download assets that have authoritative public upstream locations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import urllib.request

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
HF_TOKEN = os.environ.get("HF_TOKEN") or None

HF_ASSETS = (
    (
        "google/siglip-so400m-patch14-384",
        "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        ROOT / "minicpm_robot_track/backbones/siglip-so400m-patch14-384",
        ("model.safetensors", "config.json", "preprocessor_config.json"),
    ),
    (
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        "114c1379950215c8b35dfcd4e90a5c251dde0d32",
        ROOT / "minicpm_robot_track/backbones/dino_local_hf",
        ("model.safetensors", "config.json", "preprocessor_config.json"),
    ),
)

WHEEL_NAME = "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
WHEEL_URL = f"https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/{WHEEL_NAME}"


def download_wheel() -> None:
    destination = ROOT / "vendor" / WHEEL_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"[upstream] wheel already exists: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[upstream] downloading NVIDIA wheel: {WHEEL_URL}")
    try:
        with urllib.request.urlopen(WHEEL_URL) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print pinned sources without downloading")
    args = parser.parse_args()

    if args.dry_run:
        for repo_id, revision, destination, files in HF_ASSETS:
            print(f"{repo_id}@{revision} -> {destination}")
            for name in files:
                print(f"  {name}")
        print(f"{WHEEL_URL} -> {ROOT / 'vendor' / WHEEL_NAME}")
        return 0

    for repo_id, revision, destination, files in HF_ASSETS:
        print(f"[upstream] {repo_id}@{revision} -> {destination}")
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=destination,
                allow_patterns=list(files),
                token=HF_TOKEN,
            )
        except Exception as exc:
            if repo_id.startswith("facebook/dinov3"):
                raise RuntimeError(
                    "DINOv3 is gated. Accept its license on Hugging Face, then run "
                    "`hf auth login` or export HF_TOKEN before retrying."
                ) from exc
            raise

    download_wheel()
    print("Upstream assets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
