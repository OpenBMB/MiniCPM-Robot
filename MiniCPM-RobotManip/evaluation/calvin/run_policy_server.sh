#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
  cat <<'EOF'
Launch the MiniCPM-RobotManip WebSocket policy server for CALVIN.

Usage:
  bash run_policy_server.sh
  bash run_policy_server.sh --help

Environment:
  MINICPM_PYTHON   Python from the MiniCPM-RobotManip environment (default: python)
  CHECKPOINT       Hugging Face model ID or local checkpoint directory
                   (default: openbmb/MiniCPM-RobotManip)
  EMBODIMENT_ID    MiniCPM embodiment ID used for CALVIN requests (required)
  HOST             Listening address (default: 127.0.0.1)
  PORT             Listening port (default: 10093)
  CUDA_VISIBLE_DEVICES
                   Single visible GPU (default: 0)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MINICPM_PYTHON="${MINICPM_PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-openbmb/MiniCPM-RobotManip}"
: "${EMBODIMENT_ID:?Set EMBODIMENT_ID for the selected CALVIN checkpoint}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MINICPM_ROOT}"
exec "${MINICPM_PYTHON}" -m deployment.model_server.server_policy \
  --checkpoint "${CHECKPOINT}" \
  --device cuda \
  --host "${HOST}" \
  --port "${PORT}" \
  --default-embodiment-id "${EMBODIMENT_ID}"
