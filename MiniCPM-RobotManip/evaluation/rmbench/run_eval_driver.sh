#!/usr/bin/env bash
# Copyright 2026 The OpenBMB Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Run ONE RMBench evaluation driver (one SAPIEN env) against an already-running
# memVLA policy server. Sets up SAPIEN's Vulkan rendering, then runs the rollout
# loop in the RMBench simulator environment.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash run_eval_driver.sh [-- <extra eval_rmbench.py args>]

Environment:
  RMBENCH_PATH        External RMBench checkout (required)
  RMBENCH_PYTHON      Python from the RMBench/SAPIEN environment (default: python)
  RMBENCH_POLICY_HOST Policy server host (default: 127.0.0.1)
  RMBENCH_POLICY_PORT Policy server port (default: 10094)
  RMBENCH_GPU         GPU for this driver's rendering (default: 0)
  TASKS               Space/comma-separated task subset (default: all 10)
  SEEDS_PER_TASK      Seeds per task (default: 25)
  SEED_POOL_OFFSET    Start index into the seed schedule (default: 0)
  VIDEO_DIR           If set, record one success + one failure clip per task
  RESULTS_JSON        Where to write the structured result table
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

extra_args=()
if [[ "${1:-}" == "--" ]]; then
    shift
    extra_args=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

rmbench_path="${RMBENCH_PATH:-}"
rmbench_python="${RMBENCH_PYTHON:-python}"
host="${RMBENCH_POLICY_HOST:-127.0.0.1}"
port="${RMBENCH_POLICY_PORT:-10094}"
gpu_id="${RMBENCH_GPU:-0}"

if [[ -z "${rmbench_path}" || ! -d "${rmbench_path}" ]]; then
    echo "RMBENCH_PATH must point to an existing RMBench checkout: ${rmbench_path:-<unset>}" >&2
    exit 1
fi
if ! command -v "${rmbench_python}" >/dev/null 2>&1; then
    echo "RMBENCH_PYTHON is not executable: ${rmbench_python}" >&2
    exit 1
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "Invalid policy port: ${port}" >&2
    exit 1
fi

# --- SAPIEN Vulkan rendering (SOP three-piece setup) --------------------------
# The batch node's own /etc/vulkan/icd.d/nvidia_icd.json is sometimes broken, so
# write a known-good ICD pointing at the NVIDIA GLX loader and use it exclusively.
# The job must also run with NVIDIA_DRIVER_CAPABILITIES=all on an image that ships
# the system Vulkan loader (see README).
NVLIB=$(ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null \
    || ldconfig -p | awk '/libGLX_nvidia\.so\.0/{print $NF; exit}')
if [[ -n "${NVLIB}" ]]; then
    mkdir -p /tmp/vk_icd
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.3.242"}}\n' \
        "${NVLIB}" > /tmp/vk_icd/nvidia_icd.json
    export VK_ICD_FILENAMES=/tmp/vk_icd/nvidia_icd.json
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Normalise TASKS (accept comma or space separated) into argv.
tasks_arg=()
if [[ -n "${TASKS:-}" ]]; then
    IFS=', ' read -r -a _tasks <<< "${TASKS}"
    tasks_arg=(--tasks "${_tasks[@]}")
fi

video_arg=()
[[ -n "${VIDEO_DIR:-}" ]] && video_arg=(--video_dir "${VIDEO_DIR}")
results_arg=()
[[ -n "${RESULTS_JSON:-}" ]] && results_arg=(--results_json "${RESULTS_JSON}")

echo "[INFO] RMBench driver: gpu=${gpu_id} endpoint=${host}:${port}"
echo "[INFO] RMBENCH_PATH=${rmbench_path}"

cd "${rmbench_path}"
exec "${rmbench_python}" -m evaluation.rmbench.eval_rmbench \
    --rmbench_path "${rmbench_path}" \
    --host "${host}" \
    --port "${port}" \
    --seeds_per_task "${SEEDS_PER_TASK:-25}" \
    --seed_pool_offset "${SEED_POOL_OFFSET:-0}" \
    "${tasks_arg[@]}" \
    "${video_arg[@]}" \
    "${results_arg[@]}" \
    "${extra_args[@]}"
