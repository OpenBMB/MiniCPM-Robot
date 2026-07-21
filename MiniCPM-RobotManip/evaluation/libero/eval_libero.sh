#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
  cat <<'EOF'
Run one serial LIBERO task suite against an existing MiniCPM policy server.

Usage:
  bash eval_libero.sh [additional tyro arguments]

Required environment:
  LIBERO_HOME

Common environment:
  LIBERO_PYTHON, OUTPUT_ROOT, HOST, PORT, TASK_SUITE_NAME,
  NUM_TRIALS_PER_TASK, NUM_STEPS_WAIT, MAX_TASKS, SEED
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="${MINICPM_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MINICPM_ROOT}/outputs/evaluation/libero}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_TASKS="${MAX_TASKS:--1}"
SEED="${SEED:-7}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required and must point to the external LIBERO checkout." >&2
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0" >&2
  exit 2
fi

VIDEO_OUT_PATH="${OUTPUT_ROOT}/${TASK_SUITE_NAME}"
mkdir -p "${VIDEO_OUT_PATH}"

cd "${MINICPM_ROOT}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${MINICPM_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${EGL_DEVICE_ID}}"

exec "${LIBERO_PYTHON}" -m evaluation.libero.eval_libero \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.num-steps-wait "${NUM_STEPS_WAIT}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.seed "${SEED}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  "$@"
