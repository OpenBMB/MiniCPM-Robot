#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash run_policy_server.sh <checkpoint> [gpu_id] [port] [host] [device] [default_embodiment_id]

<checkpoint> may be a Hugging Face Hub model ID or a local checkpoint directory.

Environment defaults:
  MINICPM_PYTHON                 python
  ROBOTWIN_SERVER_GPU            0
  ROBOTWIN_SERVER_PORT           10093
  ROBOTWIN_POLICY_HOST           127.0.0.1
  MINICPM_DEVICE                 cuda
  ROBOTWIN_DEFAULT_EMBODIMENT_ID Required unless passed as argument 6
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if (( $# < 1 || $# > 6 )); then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

checkpoint="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-10093}}"
host="${4:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
device="${5:-${MINICPM_DEVICE:-cuda}}"
default_embodiment_id="${6:-${ROBOTWIN_DEFAULT_EMBODIMENT_ID:-}}"
minicpm_python="${MINICPM_PYTHON:-python}"

if [[ -e "${checkpoint}" ]]; then
    if [[ ! -d "${checkpoint}" ]]; then
        echo "Local checkpoint must be a directory: ${checkpoint}" >&2
        exit 1
    fi
    checkpoint="$(cd "${checkpoint}" && pwd -P)"
elif [[ "${checkpoint}" == /* || "${checkpoint}" == ./* || "${checkpoint}" == ../* ]]; then
    echo "Local checkpoint directory does not exist: ${checkpoint}" >&2
    exit 1
fi
if ! command -v "${minicpm_python}" >/dev/null 2>&1; then
    echo "MINICPM_PYTHON is not executable: ${minicpm_python}" >&2
    exit 1
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "Invalid server port: ${port}" >&2
    exit 1
fi
if [[ ! "${host}" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "Invalid server host: ${host}" >&2
    exit 1
fi
if [[ -z "${default_embodiment_id}" || ! "${default_embodiment_id}" =~ ^[0-9]+$ ]]; then
    echo "default_embodiment_id is required and must be a non-negative integer." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[INFO] Starting MiniCPM RoboTwin policy server"
echo "[INFO] checkpoint=${checkpoint} gpu=${gpu_id} device=${device}"
echo "[INFO] endpoint=${host}:${port} default_embodiment_id=${default_embodiment_id}"

exec "${minicpm_python}" -m deployment.model_server.server_policy \
    --checkpoint "${checkpoint}" \
    --device "${device}" \
    --host "${host}" \
    --port "${port}" \
    --default-embodiment-id "${default_embodiment_id}"
