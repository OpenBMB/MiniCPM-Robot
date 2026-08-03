#!/usr/bin/env bash
# Copyright 2026 The OpenBMB Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash run_policy_server.sh <checkpoint> [gpu_id] [port] [host] [device] [default_embodiment_id]

<checkpoint> may be a Hugging Face Hub model ID or a local checkpoint directory.

Environment defaults:
  MINICPM_PYTHON                python
  RMBENCH_SERVER_GPU            0
  RMBENCH_SERVER_PORT           10094
  RMBENCH_POLICY_HOST           127.0.0.1
  MINICPM_DEVICE                cuda
  RMBENCH_DEFAULT_EMBODIMENT_ID 4
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
gpu_id="${2:-${RMBENCH_SERVER_GPU:-0}}"
port="${3:-${RMBENCH_SERVER_PORT:-10094}}"
host="${4:-${RMBENCH_POLICY_HOST:-127.0.0.1}}"
device="${5:-${MINICPM_DEVICE:-cuda}}"
default_embodiment_id="${6:-${RMBENCH_DEFAULT_EMBODIMENT_ID:-4}}"
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
if [[ ! "${default_embodiment_id}" =~ ^[0-9]+$ ]]; then
    echo "default_embodiment_id must be a non-negative integer: ${default_embodiment_id}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Isolate the trust_remote_code dynamic-module cache per server. When several
# servers on one node load the same checkpoint concurrently they otherwise race
# writing the shared transformers_modules directory, and a loser imports a
# half-written module ("has no attribute MiniCPMVLAConfig"). A per-instance
# HF_MODULES_CACHE gives each process its own copy. Must be set before the model
# environment imports transformers.
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${TMPDIR:-/tmp}/hf_modules_${host}_${port}}"
mkdir -p "${HF_MODULES_CACHE}"

echo "[INFO] Starting MiniCPM RMBench (memVLA) policy server"
echo "[INFO] checkpoint=${checkpoint} gpu=${gpu_id} device=${device}"
echo "[INFO] endpoint=${host}:${port} default_embodiment_id=${default_embodiment_id}"

exec "${minicpm_python}" -m deployment.model_server.server_policy_memvla \
    --checkpoint "${checkpoint}" \
    --device "${device}" \
    --host "${host}" \
    --port "${port}" \
    --default-embodiment-id "${default_embodiment_id}"
