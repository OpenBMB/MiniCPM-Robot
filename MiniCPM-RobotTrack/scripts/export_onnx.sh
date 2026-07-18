#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$ROOT/realworld/trt_artifacts"

mkdir -p "$ARTIFACTS"

DINOV3_MODEL_PATH="$ROOT/minicpm_robot_track/backbones/dino_local_hf" \
DINO_ONNX_PATH="$ARTIFACTS/dino_patch_jp6_op17.onnx" \
DINO_ONNX_OPSET=17 \
python3 "$ARTIFACTS/export_dino_patch_onnx.py"

SIGLIP_MODEL_PATH="$ROOT/minicpm_robot_track/backbones/siglip-so400m-patch14-384" \
SIGLIP_POOLED_ONNX_PATH="$ARTIFACTS/siglip_pooled_jp6.onnx" \
python3 "$ARTIFACTS/export_siglip_pooled_onnx.py"

echo "ONNX files exported to $ARTIFACTS"
