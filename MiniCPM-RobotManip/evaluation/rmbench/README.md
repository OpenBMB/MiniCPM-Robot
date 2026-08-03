<!--
Copyright 2026 The OpenBMB Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# MiniCPM-RobotManip RMBench evaluation

RMBench (RoboTwin 2.0 / SAPIEN, ALOHA-AgileX dual-arm) is a suite of
memory-dependent manipulation tasks. Unlike the other benchmarks here, one request
carries the recent **history video** from three cameras: the checkpoint first
generates the current sub-task, then predicts an action chunk conditioned on the
hidden states of that same sequence. This directory drives that two-stage,
history-conditioned path.

RMBench remains an **external installation**. This directory ships only the policy
client, the continuous dense executor, the rollout driver, and the launcher.

## What is fixed by the checkpoint

Every input detail is frozen to the checkpoint's training recipe in
[`deployment/model_server/memvla_recipe.py`](../../deployment/model_server/memvla_recipe.py)
and must not be tuned:

- **Visual history** — head camera 60 frames at stride 17 (~60 s at ~1 FPS), plus
  the current frame from each wrist camera: 62 frames total.
- **Prompt** — the "aligned" style with a 17 Hz action rate, camera timestamp tags
  (`[t-X.Xs]` / `[now]`), and the unified-80D gripper-closed phrasing. The action
  head is conditioned on the hidden states of this sequence, so prompt drift
  silently degrades closed-loop behaviour.
- **Unified 80D action/state**, gripper closed mode, embodiment ID 4.

The simulator client maps RMBench's native 14D joint layout to unified 80D and back
(left joints `0:6`, right joints `17:23`, grippers `[16, 33]`); the server only ever
sees 80D.

## Environments

Use two separate Python environments:

1. `MINICPM_PYTHON` loads the checkpoint and runs the WebSocket policy server.
2. `RMBENCH_PYTHON` runs the externally installed RMBench simulator and imports the
   lightweight client from this repository.

Install the client-only wire dependencies in the RMBench environment:

```bash
/path/to/rmbench/bin/python -m pip install \
  -r MiniCPM-RobotManip/evaluation/rmbench/requirements-client.txt
```

## Installing RMBench (external)

```bash
git clone https://github.com/RoboTwin-Platform/RMBench
# Download the heavy assets + vendored curobo data (gitignored in the checkout):
modelscope download --dataset keithyc/RMBench_sim --local_dir RMBench
export RMBENCH_PATH=/path/to/RMBench
export MINICPM_PYTHON=/path/to/minicpm/bin/python
export RMBENCH_PYTHON=/path/to/rmbench/bin/python
```

After downloading assets, patch the curobo YAMLs whose collection-machine paths are
hardcoded, replacing them with your checkout path:

```bash
find "$RMBENCH_PATH" -name 'curobo_*.yml' -not -name '*_tmp.yml' \
  -exec sed -i "s|/data/.*/RMBench|$RMBENCH_PATH|g" {} +
```

## Rendering requirements (closed-loop, all three needed)

SAPIEN rendering in a batch job needs, together:

1. **An image with the system Vulkan loader** (a training image without it fails at
   `failed to find a rendering device`).
2. **`NVIDIA_DRIVER_CAPABILITIES=all`** on the job.
3. **A known-good Vulkan ICD** — `run_eval_driver.sh` writes one to
   `/tmp/vk_icd/nvidia_icd.json` and points `VK_ICD_FILENAMES` at it, because the
   node's own `/etc/vulkan/icd.d/nvidia_icd.json` is sometimes broken.

If your GPU architecture differs from the prebuilt curobo `.so` (e.g. sm_90 vs
sm_80), the driver JIT may raise `provided PTX was compiled with an unsupported
toolchain`; check with `cuobjdump --list-elf <a curobo .so>` and rebuild curobo for
your architecture. On sm_90 and newer, curobo's L-BFGS kernel can also surface
undefined behaviour that is tolerated on sm_80.

## Multi-GPU evaluation

From any directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RMBENCH_PATH=/path/to/RMBench \
MINICPM_PYTHON=/path/to/minicpm/bin/python \
RMBENCH_PYTHON=/path/to/rmbench/bin/python \
bash MiniCPM-RobotManip/evaluation/rmbench/start_eval.sh \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --seeds-per-task 25
```

`start_eval.sh` starts one policy server and one driver per GPU, shards the 10 tasks
across GPUs, waits for each server's history/subtask handshake, and merges per-GPU
results into `outputs/evaluation/rmbench/<timestamp>_<pid>/summary.json`. The summary
reports **macro9** (the reported RMBench metric, excluding `place_block_mat`) and
macro10.

`--checkpoint` accepts a Hugging Face Hub model ID or a local checkpoint directory.

## Manual single-task evaluation

Start the server in the MiniCPM environment:

```bash
MINICPM_PYTHON=/path/to/minicpm/bin/python \
bash MiniCPM-RobotManip/evaluation/rmbench/run_policy_server.sh \
  openbmb/MiniCPM-RobotManip 0 10094
```

Then run the driver in the RMBench environment:

```bash
RMBENCH_PATH=/path/to/RMBench \
RMBENCH_PYTHON=/path/to/rmbench/bin/python \
TASKS=observe_and_pickup SEEDS_PER_TASK=2 \
bash MiniCPM-RobotManip/evaluation/rmbench/run_eval_driver.sh
```

## Seeds

Evaluation uses **held-out** seeds starting at `base_seed=100000`, disjoint from the
small collection seeds. By default each seed is validated expert-solvable through
RMBench's CuRobo solve, which also generates the task language. That solve runs live
on every seed (a few seconds each) — it is the dominant fixed cost of a run, but the
`base_seed=100000` schedule is deterministic, so results are reproducible.

## Policy contract

- Camera order: `[head, left_wrist, right_wrist]` (`cam_high`, `cam_left_wrist`,
  `cam_right_wrist`), transported as `examples[0].views` (per-camera frame lists).
- Client image size: 448 x 448.
- State: measured 14D joints packed into unified 80D.
- Response: `data.actions` shaped `(1, 30, 80)` plus `data.subtask` text.
- Executor: continuous dense (`dense_substeps=15`, one dataset frame = 15 physics
  steps), NOT the upstream rest-to-rest `take_action`.
- Embodiment: ID 4.

## Metric

macro9 = mean success rate over 9 tasks (all 10 except `place_block_mat`), to match
the RMBench figure other models report. Noise scale: 25 seeds gives about ±8–10pp per
task and ±4pp on macro9; use 100 seeds for a SOTA claim or a sub-4pp comparison.
