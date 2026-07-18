#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds/install}"
failures=0

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        printf 'OK   command: %s\n' "$1"
    else
        printf 'FAIL command: %s\n' "$1"
        failures=$((failures + 1))
    fi
}

check_file() {
    if [[ -e "$1" ]]; then
        printf 'OK   file: %s\n' "$1"
    else
        printf 'FAIL file: %s\n' "$1"
        failures=$((failures + 1))
    fi
}

check_any_file() {
    local label=$1
    shift
    local path
    for path in "$@"; do
        if [[ -e "$path" ]]; then
            printf 'OK   %s: %s\n' "$label" "$path"
            return
        fi
    done
    printf 'FAIL %s: none of the expected files exists\n' "$label"
    failures=$((failures + 1))
}

check_command python3
check_command nvpmodel
check_command jetson_clocks
check_command ros2
check_file /etc/nv_tegra_release
check_file /usr/src/tensorrt/bin/trtexec
check_file /opt/ros/humble/setup.bash
check_file "$CYCLONEDDS_HOME/lib"
check_file "$ROOT/minicpm_robot_track/backbones/dino_local_hf/model.safetensors"
check_file "$ROOT/minicpm_robot_track/backbones/siglip-so400m-patch14-384/model.safetensors"
MODEL_DIR="$ROOT/minicpm_robot_track/checkpoints/MiniCPM-RobotTrack"
export MODEL_DIR
check_file "$MODEL_DIR/config.json"
check_file "$MODEL_DIR/configuration_robottrack.py"
check_file "$MODEL_DIR/configuration_minicpm.py"
check_file "$MODEL_DIR/modeling_robottrack.py"
check_file "$MODEL_DIR/modeling_minicpm.py"
check_file "$MODEL_DIR/tokenizer_config.json"
check_any_file "model weights" \
    "$MODEL_DIR/model.safetensors" \
    "$MODEL_DIR/model.safetensors.index.json" \
    "$MODEL_DIR/pytorch_model.bin" \
    "$MODEL_DIR/pytorch_model.bin.index.json"
check_file "$ROOT/realworld/trt_artifacts/dino_patch_target_fp16.engine"
check_file "$ROOT/realworld/trt_artifacts/siglip_pooled_target_maxn_fp16.engine"

python3 - <<'PY' || failures=$((failures + 1))
import os

modules = ("torch", "torchvision", "tensorrt", "transformers", "cv2", "yaml")
for name in modules:
    module = __import__(name)
    print(f"OK   python: {name} {getattr(module, '__version__', '')}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("FAIL torch.cuda.is_available() is false")
print("OK   CUDA:", torch.cuda.get_device_name(0))

from transformers import AutoConfig, AutoTokenizer

model_dir = os.environ["MODEL_DIR"]
config = AutoConfig.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
print("OK   Hugging Face config:", config.model_type)
print("OK   Hugging Face tokenizer:", type(tokenizer).__name__)
PY

if (( failures > 0 )); then
    echo "Preflight failed: $failures check(s) failed." >&2
    exit 1
fi
echo "Preflight passed."
