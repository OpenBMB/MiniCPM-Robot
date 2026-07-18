#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ARTIFACTS="$ROOT/realworld/trt_artifacts"
LOGS="$ROOT/logs"

[[ -x "$TRTEXEC" ]] || { echo "trtexec not found: $TRTEXEC" >&2; exit 1; }
[[ -f "$ARTIFACTS/dino_patch_jp6_op17.onnx" ]] || { echo "DINO ONNX is missing" >&2; exit 1; }
[[ -f "$ARTIFACTS/siglip_pooled_jp6.onnx" ]] || { echo "SigLIP ONNX is missing" >&2; exit 1; }

mkdir -p "$LOGS"

echo "Power mode before build:"
nvpmodel -q
echo "Build engines only after MAXN mode 0 and jetson_clocks are active."

"$TRTEXEC" \
    --onnx="$ARTIFACTS/dino_patch_jp6_op17.onnx" \
    --saveEngine="$ARTIFACTS/dino_patch_target_fp16.engine" \
    --fp16 --memPoolSize=workspace:4096MiB \
    --builderOptimizationLevel=5 --skipInference \
    2>&1 | tee "$LOGS/build-dino.log"

"$TRTEXEC" \
    --onnx="$ARTIFACTS/siglip_pooled_jp6.onnx" \
    --saveEngine="$ARTIFACTS/siglip_pooled_target_maxn_fp16.engine" \
    --fp16 --memPoolSize=workspace:4096MiB \
    --builderOptimizationLevel=5 --skipInference \
    2>&1 | tee "$LOGS/build-siglip.log"

echo "TensorRT engines built in $ARTIFACTS"
