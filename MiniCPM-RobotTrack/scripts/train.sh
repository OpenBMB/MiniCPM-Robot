#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

CHECKPOINT=${1:-minicpm_robot_track/checkpoints/MiniCPM-RobotTrack}

python -m minicpm_robot_track.training \
  --init-checkpoint "$CHECKPOINT" \
  --stt-json sim_data/train/stt/jsonl \
  --stt-cache sim_data/train/stt/vision_cache \
  --at-json sim_data/train/at/jsonl \
  --at-cache sim_data/train/at/vision_cache \
  --dt-json sim_data/train/dt/jsonl \
  --dt-cache sim_data/train/dt/vision_cache \
  --output-dir outputs/train \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --head-lr 1e-4 \
  --control-query-lr 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --min-lr-ratio 0.1 \
  --gradient-checkpointing
