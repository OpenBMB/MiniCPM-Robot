#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

CHECKPOINT=${1:-minicpm_robot_track/checkpoints/MiniCPM-RobotTrack}
OUTPUT=${2:-results/at}
SPLIT_COUNT=30

for ((SPLIT_ID = 0; SPLIT_ID <= SPLIT_COUNT; SPLIT_ID++)); do
  ARGS=(
    --task at
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT"
    --split-count "$SPLIT_COUNT"
    --split-id "$SPLIT_ID"
  )
  if [[ -f "$CHECKPOINT" && -n "${MINICPM_MODEL_PATH:-}" ]]; then
    ARGS+=(--backbone "$MINICPM_MODEL_PATH")
  fi
  python -m minicpm_robot_track.evaluation.runner "${ARGS[@]}"
done
