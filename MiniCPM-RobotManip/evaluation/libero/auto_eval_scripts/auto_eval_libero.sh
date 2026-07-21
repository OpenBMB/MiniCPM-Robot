#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  auto_eval_libero.sh [--dry-run] [--checkpoint ID_OR_DIR ...]
  auto_eval_libero.sh [--dry-run] [ID_OR_DIR ...]

Runs the checkpoint × task-suite matrix with one bounded worker per fixed GPU
slot. A checkpoint may be a Hugging Face model ID or local checkpoint
directory.

Checkpoint selection precedence:
  1. Positional arguments or repeated --checkpoint options
  2. CHECKPOINTS (space-separated; use CLI for paths containing spaces)
  3. CHECKPOINT
  4. CKPT_DIR immediate child directories
  5. openbmb/MiniCPM-RobotManip

Environment:
  LIBERO_HOME        External LIBERO checkout (required unless --dry-run)
  MINICPM_PYTHON     Model-server Python (default: python)
  LIBERO_PYTHON      LIBERO Python (default: python)
  OUTPUT_ROOT        Output base (default: outputs/evaluation/libero)
  TASK_SUITES        Space/comma-separated suites
                     (default: libero_10 libero_goal libero_object libero_spatial)
  GPU_LIST           Space/comma-separated fixed slots (default: 0)
  EGL_DEVICE_LIST    EGL device per GPU slot (default: GPU_LIST)
  HOST               Server host (default: 127.0.0.1)
  BASE_PORT          Port for slot 0; slot N uses BASE_PORT+N (default: 10093)
  SERVER_TIMEOUT     Metadata/ping readiness timeout (default: 300)
  EMBODIMENT_ID      MiniCPM default embodiment ID (required)
  RUN_ID             Unique output run name (default: UTC timestamp + PID)
  NUM_TRIALS_PER_TASK, NUM_STEPS_WAIT, MAX_TASKS, SEED

Examples:
  LIBERO_HOME=/opt/LIBERO GPU_LIST="0 1" EMBODIMENT_ID=0 \
    bash auto_eval_libero.sh --checkpoint openbmb/MiniCPM-RobotManip

  LIBERO_HOME=/opt/LIBERO TASK_SUITES=libero_goal EMBODIMENT_ID=0 \
    bash auto_eval_libero.sh --checkpoint /models/minicpm-libero

  GPU_LIST="0 1" EMBODIMENT_ID=0 bash auto_eval_libero.sh --dry-run \
    --checkpoint openbmb/MiniCPM-RobotManip
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="${MINICPM_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
WORKER_SCRIPT="${SCRIPT_DIR}/eval_libero_parallel.sh"

dry_run=0
declare -a cli_checkpoints=()
while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --checkpoint)
      if (($# < 2)); then
        echo "--checkpoint requires a Hugging Face ID or local path." >&2
        exit 2
      fi
      cli_checkpoints+=("$2")
      shift 2
      ;;
    --checkpoint=*)
      cli_checkpoints+=("${1#*=}")
      shift
      ;;
    --)
      shift
      while (($# > 0)); do
        cli_checkpoints+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      cli_checkpoints+=("$1")
      shift
      ;;
  esac
done

MINICPM_PYTHON="${MINICPM_PYTHON:-python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
LIBERO_HOME="${LIBERO_HOME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MINICPM_ROOT}/outputs/evaluation/libero}"
TASK_SUITES="${TASK_SUITES:-libero_10 libero_goal libero_object libero_spatial}"
GPU_LIST="${GPU_LIST:-0}"
EGL_DEVICE_LIST="${EGL_DEVICE_LIST:-${GPU_LIST}}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-10093}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-360}"
EMBODIMENT_ID="${EMBODIMENT_ID:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_TASKS="${MAX_TASKS:--1}"
SEED="${SEED:-7}"

declare -a checkpoints=()
if ((${#cli_checkpoints[@]} > 0)); then
  checkpoints=("${cli_checkpoints[@]}")
elif [[ -n "${CHECKPOINTS:-}" ]]; then
  read -r -a checkpoints <<<"${CHECKPOINTS}"
elif [[ -n "${CHECKPOINT:-}" ]]; then
  checkpoints=("${CHECKPOINT}")
elif [[ -n "${CKPT_DIR:-}" ]]; then
  shopt -s nullglob
  for candidate in "${CKPT_DIR}"/*/; do
    if [[ -d "${candidate}" ]]; then
      checkpoints+=("${candidate%/}")
    fi
  done
  shopt -u nullglob
else
  checkpoints=("openbmb/MiniCPM-RobotManip")
fi

task_suite_text="${TASK_SUITES//,/ }"
gpu_list_text="${GPU_LIST//,/ }"
egl_device_list_text="${EGL_DEVICE_LIST//,/ }"
read -r -a task_suites <<<"${task_suite_text}"
read -r -a gpu_slots <<<"${gpu_list_text}"
read -r -a egl_slots <<<"${egl_device_list_text}"

if ((${#checkpoints[@]} == 0)); then
  echo "No checkpoints were selected." >&2
  exit 2
fi
if ((${#task_suites[@]} == 0)); then
  echo "TASK_SUITES must contain at least one suite." >&2
  exit 2
fi
if ((${#gpu_slots[@]} == 0)); then
  echo "GPU_LIST must contain at least one GPU." >&2
  exit 2
fi
if ((${#egl_slots[@]} != ${#gpu_slots[@]})); then
  echo "EGL_DEVICE_LIST must contain exactly one entry per GPU_LIST slot." >&2
  exit 2
fi
if [[ ! "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "BASE_PORT must be an integer; got '${BASE_PORT}'." >&2
  exit 2
fi
if [[ ! "${SERVER_TIMEOUT}" =~ ^[0-9]+$ ]] || ((SERVER_TIMEOUT < 1)); then
  echo "SERVER_TIMEOUT must be a positive integer." >&2
  exit 2
fi
if [[ -z "${EMBODIMENT_ID}" || ! "${EMBODIMENT_ID}" =~ ^[0-9]+$ ]]; then
  echo "EMBODIMENT_ID is required and must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

num_slots=${#gpu_slots[@]}
if ((BASE_PORT < 1 || BASE_PORT + num_slots - 1 > 65535)); then
  echo "The fixed slot ports must all be in [1, 65535]." >&2
  exit 2
fi
for ((left = 0; left < num_slots; left++)); do
  if [[ -z "${gpu_slots[$left]}" ]]; then
    echo "GPU_LIST contains an empty slot." >&2
    exit 2
  fi
  if [[ -z "${egl_slots[$left]}" ]]; then
    echo "EGL_DEVICE_LIST contains an empty slot." >&2
    exit 2
  fi
  for ((right = left + 1; right < num_slots; right++)); do
    if [[ "${gpu_slots[$left]}" == "${gpu_slots[$right]}" ]]; then
      echo "GPU_LIST contains duplicate GPU '${gpu_slots[$left]}'." >&2
      exit 2
    fi
  done
done

declare -a job_checkpoints=()
declare -a job_suites=()
for checkpoint in "${checkpoints[@]}"; do
  if [[ -z "${checkpoint}" ]]; then
    echo "Checkpoint values must be non-empty." >&2
    exit 2
  fi
  if [[ -e "${checkpoint}" && ! -d "${checkpoint}" ]]; then
    echo "Local MiniCPM checkpoint must be a directory: ${checkpoint}" >&2
    exit 2
  fi
  if [[ ! -e "${checkpoint}" ]] \
    && [[ "${checkpoint}" == /* || "${checkpoint}" == ./* || "${checkpoint}" == ../* ]]; then
    echo "Local checkpoint directory does not exist: ${checkpoint}" >&2
    exit 2
  fi
  for task_suite in "${task_suites[@]}"; do
    job_checkpoints+=("${checkpoint}")
    job_suites+=("${task_suite}")
  done
done
num_jobs=${#job_checkpoints[@]}

echo "=========================================="
echo " MiniCPM LIBERO auto evaluation"
echo "=========================================="
echo " Run ID      : ${RUN_ID}"
echo " Checkpoints : ${checkpoints[*]}"
echo " Task suites : ${task_suites[*]}"
echo " GPU slots   : ${gpu_slots[*]}"
echo " EGL devices : ${egl_slots[*]}"
echo " Slot ports  : ${BASE_PORT}..$((BASE_PORT + num_slots - 1))"
echo " Output root : ${OUTPUT_ROOT}"
echo " Jobs        : ${num_jobs}"
echo "=========================================="

if ((dry_run == 1)); then
  for ((job_id = 0; job_id < num_jobs; job_id++)); do
    slot=$((job_id % num_slots))
    printf '[dry-run job %d] slot=%d gpu=%s egl=%s port=%d suite=%s checkpoint=%s\n' \
      "${job_id}" \
      "${slot}" \
      "${gpu_slots[$slot]}" \
      "${egl_slots[$slot]}" \
      "$((BASE_PORT + slot))" \
      "${job_suites[$job_id]}" \
      "${job_checkpoints[$job_id]}"
  done
  exit 0
fi

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required and must point to the external LIBERO checkout." >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
SCHEDULE_MANIFEST="${RUN_ROOT}/manifest.tsv"
STATUS_MANIFEST="${RUN_ROOT}/status.tsv"
mkdir -p "${RUN_ROOT}"
{
  printf 'job_id\tslot\tgpu\tegl_device\tport\ttask_suite\tcheckpoint\n'
  for ((job_id = 0; job_id < num_jobs; job_id++)); do
    slot=$((job_id % num_slots))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${job_id}" \
      "${slot}" \
      "${gpu_slots[$slot]}" \
      "${egl_slots[$slot]}" \
      "$((BASE_PORT + slot))" \
      "${job_suites[$job_id]}" \
      "${job_checkpoints[$job_id]}"
  done
} >"${SCHEDULE_MANIFEST}"
printf 'job_id\texit_status\n' >"${STATUS_MANIFEST}"

declare -a slot_pids=()
declare -a slot_job_ids=()
declare -a slot_next_jobs=()
for ((slot = 0; slot < num_slots; slot++)); do
  slot_pids[$slot]=""
  slot_job_ids[$slot]=""
  slot_next_jobs[$slot]="${slot}"
done

cleanup_workers() {
  local status=$?
  local pid
  local deadline
  local any_running

  trap - EXIT INT TERM HUP
  for ((slot = 0; slot < num_slots; slot++)); do
    pid="${slot_pids[$slot]:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + 15))
  while ((SECONDS < deadline)); do
    any_running=0
    for ((slot = 0; slot < num_slots; slot++)); do
      pid="${slot_pids[$slot]:-}"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        any_running=1
      fi
    done
    if ((any_running == 0)); then
      break
    fi
    sleep 0.1
  done

  for ((slot = 0; slot < num_slots; slot++)); do
    pid="${slot_pids[$slot]:-}"
    if [[ -n "${pid}" ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
      fi
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

handle_signal() {
  local signal_name="$1"
  case "${signal_name}" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    HUP) exit 129 ;;
  esac
}

trap cleanup_workers EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP

remaining_jobs=${num_jobs}
failed_jobs=0
while ((remaining_jobs > 0)); do
  for ((slot = 0; slot < num_slots; slot++)); do
    pid="${slot_pids[$slot]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      completed_job="${slot_job_ids[$slot]}"
      if wait "${pid}"; then
        worker_status=0
      else
        worker_status=$?
      fi
      printf '%s\t%s\n' "${completed_job}" "${worker_status}" >>"${STATUS_MANIFEST}"
      if ((worker_status != 0)); then
        failed_jobs=$((failed_jobs + 1))
        echo "[job ${completed_job}] failed with status ${worker_status}." >&2
      else
        echo "[job ${completed_job}] completed."
      fi
      slot_pids[$slot]=""
      slot_job_ids[$slot]=""
      remaining_jobs=$((remaining_jobs - 1))
    fi

    if [[ -z "${slot_pids[$slot]:-}" ]]; then
      next_job="${slot_next_jobs[$slot]}"
      if ((next_job < num_jobs)); then
        gpu_id="${gpu_slots[$slot]}"
        egl_device_id="${egl_slots[$slot]}"
        port=$((BASE_PORT + slot))
        checkpoint="${job_checkpoints[$next_job]}"
        task_suite="${job_suites[$next_job]}"
        echo "[job ${next_job}] slot=${slot} gpu=${gpu_id} egl=${egl_device_id} port=${port} suite=${task_suite}"

        env \
          MINICPM_ROOT="${MINICPM_ROOT}" \
          MINICPM_PYTHON="${MINICPM_PYTHON}" \
          LIBERO_HOME="${LIBERO_HOME}" \
          LIBERO_PYTHON="${LIBERO_PYTHON}" \
          OUTPUT_ROOT="${OUTPUT_ROOT}" \
          RUN_ID="${RUN_ID}" \
          JOB_ID="${next_job}" \
          HOST="${HOST}" \
          SERVER_TIMEOUT="${SERVER_TIMEOUT}" \
          EMBODIMENT_ID="${EMBODIMENT_ID}" \
          NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
          NUM_STEPS_WAIT="${NUM_STEPS_WAIT}" \
          MAX_TASKS="${MAX_TASKS}" \
          SEED="${SEED}" \
          EGL_DEVICE_ID="${egl_device_id}" \
          MUJOCO_EGL_DEVICE_ID="${egl_device_id}" \
          bash "${WORKER_SCRIPT}" \
          "${checkpoint}" "${task_suite}" "${gpu_id}" "${port}" &
        slot_pids[$slot]=$!
        slot_job_ids[$slot]="${next_job}"
        slot_next_jobs[$slot]=$((next_job + num_slots))
      fi
    fi
  done

  if ((remaining_jobs > 0)); then
    sleep 0.2
  fi
done

if ((failed_jobs > 0)); then
  echo "${failed_jobs} of ${num_jobs} LIBERO jobs failed. See ${STATUS_MANIFEST}." >&2
  exit 1
fi

echo "All ${num_jobs} LIBERO jobs completed successfully."
echo "Outputs: ${RUN_ROOT}"
