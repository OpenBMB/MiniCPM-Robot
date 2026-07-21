#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
  cat <<'EOF'
Launch the MiniCPM-RobotManip policy server for LIBERO.

Usage:
  bash run_policy_server.sh [additional server arguments]

Environment:
  MINICPM_PYTHON  Model-server Python (default: python)
  CHECKPOINT      Hub ID or local checkpoint directory
  EMBODIMENT_ID   LIBERO embodiment ID (required)
  HOST            Listening host (default: 127.0.0.1)
  PORT            Listening port (default: 10093)
  GPU_ID          CUDA device exposed to the server (default: 0)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="${MINICPM_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MINICPM_PYTHON="${MINICPM_PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-openbmb/MiniCPM-RobotManip}"
: "${EMBODIMENT_ID:?Set EMBODIMENT_ID for the selected LIBERO checkpoint}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
GPU_ID="${GPU_ID:-0}"

cd "${MINICPM_ROOT}"
export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${MINICPM_PYTHON}" -m deployment.model_server.server_policy \
  --checkpoint "${CHECKPOINT}" \
  --device cuda \
  --host "${HOST}" \
  --port "${PORT}" \
  --default-embodiment-id "${EMBODIMENT_ID}" \
  "$@"
