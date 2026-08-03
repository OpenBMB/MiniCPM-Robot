#!/usr/bin/env bash
# Copyright 2026 The OpenBMB Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Multi-GPU RMBench closed-loop evaluation. Starts one memVLA policy server and
# one evaluation driver per GPU (the model and simulator run in SEPARATE Python
# environments), shards the task list across GPUs, then merges per-GPU results
# into one macro-average summary.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash start_eval.sh --checkpoint <ckpt> [options] [task ...]

Required:
  --checkpoint <ckpt>       HF model ID or local checkpoint directory

Options:
  --seeds-per-task <N>      Seeds per task (default: 25)
  --base-port <PORT>        First policy-server port (default: 10094)
  --server-timeout <SEC>    Seconds to wait for each server to become ready (default: 1800)
  --video                   Record one success + one failure clip per task
  --output-root <DIR>       Output root (default: <repo>/outputs/evaluation/rmbench)
  [task ...]                Task subset (default: all 10)

Environment:
  MINICPM_PYTHON            Python for the model server (required)
  RMBENCH_PYTHON            Python for the RMBench/SAPIEN simulator (required)
  RMBENCH_PATH              External RMBench checkout (required)
  CUDA_VISIBLE_DEVICES      GPU list; falls back to nvidia-smi
EOF
}

checkpoint=""
seeds_per_task=25
base_port=10094
server_timeout=1800
want_video=0
output_root=""
tasks=()

while (( $# > 0 )); do
    case "$1" in
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --seeds-per-task) seeds_per_task="$2"; shift 2 ;;
        --base-port) base_port="$2"; shift 2 ;;
        --server-timeout) server_timeout="$2"; shift 2 ;;
        --video) want_video=1; shift ;;
        --output-root) output_root="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) tasks+=("$1"); shift ;;
    esac
done

if [[ -z "${checkpoint}" ]]; then
    echo "--checkpoint is required" >&2
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
minicpm_python="${MINICPM_PYTHON:-python}"
rmbench_python="${RMBENCH_PYTHON:-python}"
rmbench_path="${RMBENCH_PATH:-}"

for check in "MINICPM_PYTHON=${minicpm_python}" "RMBENCH_PYTHON=${rmbench_python}"; do
    if ! command -v "${check#*=}" >/dev/null 2>&1; then
        echo "${check%%=*} is not executable: ${check#*=}" >&2
        exit 1
    fi
done
if [[ -z "${rmbench_path}" || ! -d "${rmbench_path}" ]]; then
    echo "RMBENCH_PATH must point to an existing RMBench checkout: ${rmbench_path:-<unset>}" >&2
    exit 1
fi

# GPU discovery: CUDA_VISIBLE_DEVICES first, then nvidia-smi.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi
if (( ${#gpus[@]} == 0 )); then
    echo "No GPUs found (set CUDA_VISIBLE_DEVICES)." >&2
    exit 1
fi
num_gpus=${#gpus[@]}

DEFAULT_TASKS=(observe_and_pickup put_back_block swap_T press_button place_block_mat
               battery_try rearrange_blocks swap_blocks cover_blocks blocks_ranking_try)
if (( ${#tasks[@]} == 0 )); then
    tasks=("${DEFAULT_TASKS[@]}")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_root}" ]]; then
    output_root="${MINICPM_ROOT}/outputs/evaluation/rmbench"
fi
run_dir="${output_root}/${timestamp}_$$"
mkdir -p "${run_dir}"
echo "[INFO] run dir: ${run_dir}"
echo "[INFO] GPUs: ${gpus[*]} | tasks: ${#tasks[@]} | seeds/task: ${seeds_per_task}"

server_pids=()
driver_pids=()
result_files=()

cleanup() {
    for pid in "${driver_pids[@]}" "${server_pids[@]}"; do
        [[ -n "${pid}" ]] && kill -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# Split by SEED, not by task: every GPU runs all tasks over a disjoint slice of the
# per-task seed schedule. This spreads each task's long tail across all GPUs, so no
# GPU idles waiting on one slow task (task-sharding left the longest task on a lone
# GPU while the rest finished). Results are pooled per task at the end.
all_tasks="${tasks[*]}"
active_gpus=$(( num_gpus < seeds_per_task ? num_gpus : seeds_per_task ))
(( active_gpus < 1 )) && active_gpus=1
base_share=$(( seeds_per_task / active_gpus ))
rem_share=$(( seeds_per_task % active_gpus ))

declare -a share offset
acc=0
for (( i = 0; i < num_gpus; i++ )); do
    if (( i < active_gpus )); then
        share[i]=$(( base_share + (i < rem_share ? 1 : 0) ))
    else
        share[i]=0
    fi
    offset[i]=$acc
    acc=$(( acc + share[i] ))
done

for (( i = 0; i < num_gpus; i++ )); do
    (( share[i] == 0 )) && continue
    gpu="${gpus[i]}"
    port=$(( base_port + i ))

    server_log="${run_dir}/server_gpu${gpu}.log"
    echo "[INFO] GPU ${gpu}: starting server on :${port} (log: ${server_log})"
    setsid bash "${SCRIPT_DIR}/run_policy_server.sh" "${checkpoint}" "${gpu}" "${port}" \
        > "${server_log}" 2>&1 &
    server_pids+=("$!")
done

# Wait for every server to answer a metadata handshake.
wait_ready() {
    local host="127.0.0.1" port="$1" deadline=$(( SECONDS + server_timeout ))
    while (( SECONDS < deadline )); do
        if MINICPM_ROOT="${MINICPM_ROOT}" "${minicpm_python}" - "${host}" "${port}" <<'PY'
import sys
sys.path.insert(0, __import__("os").environ["MINICPM_ROOT"])
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
host, port = sys.argv[1], int(sys.argv[2])
try:
    client = WebsocketClientPolicy(host, port, open_timeout=5)
    meta = client.get_server_metadata()
    client.close()
    caps = meta.get("capabilities", {})
    sys.exit(0 if caps.get("history") and caps.get("subtask") else 3)
except Exception:
    sys.exit(1)
PY
        then
            return 0
        fi
        sleep 5
    done
    return 1
}

for (( i = 0; i < num_gpus; i++ )); do
    (( share[i] == 0 )) && continue
    gpu="${gpus[i]}"
    port=$(( base_port + i ))
    if ! wait_ready "${port}"; then
        echo "[ERROR] server on :${port} (GPU ${gpu}) not ready within ${server_timeout}s" >&2
        exit 1
    fi
    echo "[INFO] GPU ${gpu}: server ready on :${port}"
done

# Start one driver per GPU: all tasks, a disjoint seed slice each.
for (( i = 0; i < num_gpus; i++ )); do
    (( share[i] == 0 )) && continue
    gpu="${gpus[i]}"
    port=$(( base_port + i ))
    driver_log="${run_dir}/driver_gpu${gpu}.log"
    result_json="${run_dir}/results_gpu${gpu}.json"
    result_files+=("${result_json}")
    video_dir=""
    (( want_video )) && video_dir="${run_dir}/videos/gpu${gpu}"

    echo "[INFO] GPU ${gpu}: driver — ${share[i]} seeds/task from offset ${offset[i]} (log: ${driver_log})"
    setsid env \
        RMBENCH_PATH="${rmbench_path}" \
        RMBENCH_PYTHON="${rmbench_python}" \
        RMBENCH_POLICY_HOST="127.0.0.1" \
        RMBENCH_POLICY_PORT="${port}" \
        RMBENCH_GPU="${gpu}" \
        TASKS="${all_tasks}" \
        SEEDS_PER_TASK="${share[i]}" \
        SEED_POOL_OFFSET="${offset[i]}" \
        VIDEO_DIR="${video_dir}" \
        RESULTS_JSON="${result_json}" \
        bash "${SCRIPT_DIR}/run_eval_driver.sh" \
        > "${driver_log}" 2>&1 &
    driver_pids+=("$!")
done

# Wait for all drivers.
fail=0
for pid in "${driver_pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

# Merge per-GPU result tables.
summary_json="${run_dir}/summary.json"
"${minicpm_python}" "${SCRIPT_DIR}/merge_results.py" --out "${summary_json}" "${result_files[@]}" || fail=1

echo "[INFO] summary: ${summary_json}"
exit "${fail}"
