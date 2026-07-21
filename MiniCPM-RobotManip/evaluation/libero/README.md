# Standard LIBERO evaluation

This directory ports the standard starVLA LIBERO evaluator from commit
`631aae02afe6d95876e923ff518e8ff2ab9a2f88`. The derived source files retain
the starVLA MIT attribution and identify OpenBMB modifications. OpenPI,
LIBERO-plus, training code, simulator assets, success-rate scanners, and video
cleanup helpers are intentionally excluded.

## Environments

Use two Python environments:

1. `MINICPM_PYTHON`: the MiniCPM-RobotManip model/server environment.
2. `LIBERO_PYTHON`: an environment containing an externally installed LIBERO
   checkout plus the evaluator-side packages:

```bash
/path/to/libero/python -m pip install \
  -r evaluation/libero/requirements-client.txt
```

`LIBERO_HOME` must point to the external LIBERO checkout. The launchers resolve
`MINICPM_ROOT` from their own location, so they can be called from any working
directory.

## Single server and evaluator

Start the model server in one terminal:

```bash
CHECKPOINT=openbmb/MiniCPM-RobotManip \
EMBODIMENT_ID=0 \
MINICPM_PYTHON=/path/to/minicpm/python \
bash evaluation/libero/run_policy_server.sh
```

Run LIBERO in another terminal:

```bash
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/python \
OUTPUT_ROOT=/path/to/outputs/evaluation/libero/manual \
TASK_SUITE_NAME=libero_goal \
bash evaluation/libero/eval_libero.sh
```

Both scripts default to `127.0.0.1:10093`. Useful smoke-test variables are
`MAX_TASKS=1` and `NUM_TRIALS_PER_TASK=1`. The evaluator never loads a
checkpoint and never derives output paths from one.

## Bounded multi-GPU auto-eval

The auto-evaluator creates fixed GPU slots. Slot `N` always uses
`GPU_LIST[N]` and `BASE_PORT + N`, and runs at most one server/evaluator job at
a time. Different slots run concurrently.

`auto_eval_libero.sh` keeps the top-level slot scheduler. Each slot invokes the
thin `eval_libero_parallel.sh` entry point; server readiness, process groups,
signals, and per-job manifests are implemented in `parallel_worker.py`.

```bash
LIBERO_HOME=/path/to/LIBERO \
MINICPM_PYTHON=/path/to/minicpm/python \
LIBERO_PYTHON=/path/to/libero/python \
GPU_LIST="0 1" \
EGL_DEVICE_LIST="0 1" \
EMBODIMENT_ID=0 \
TASK_SUITES="libero_10 libero_goal libero_object libero_spatial" \
OUTPUT_ROOT=/path/to/outputs/evaluation/libero \
bash evaluation/libero/auto_eval_scripts/auto_eval_libero.sh \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --checkpoint /path/to/local/checkpoint-directory
```

Inspect configuration without launching a model or simulator:

```bash
EMBODIMENT_ID=0 bash evaluation/libero/auto_eval_scripts/auto_eval_libero.sh \
  --dry-run --checkpoint openbmb/MiniCPM-RobotManip
```

`EGL_DEVICE_LIST` maps one physical EGL device to each `GPU_LIST` slot. It
defaults to the same list but can be overridden on systems whose EGL and CUDA
device numbering differ.

Use `--help` for all environment variables. Each run writes scheduling/status
manifests and per-job server/evaluator logs below
`OUTPUT_ROOT/RUN_ID/`. Readiness requires the server process to remain alive
and the repository WebSocket client to complete both metadata validation and a
protocol ping. Any failed job makes the top-level command return nonzero.
`EXIT`, `INT`, `TERM`, and `HUP` handlers terminate and reap active evaluators
and servers.

## Evaluation contract

- Camera order is `[agentview, eye_in_hand]`.
- Both LIBERO images are rotated 180 degrees, then resized to 448×448.
- Proprioceptive state is not sent; MiniCPM uses its existing 80-D zero state.
- Requests contain only `examples=[{"image": ..., "lang": ...}]`; no
  normalization, un-normalization, DDIM, sampling, or CFG fields are sent.
- Responses must report `ok=true` and contain finite floating actions with
  shape `(1, T, D)`, where `D >= 7`.
- Model action values remain unchanged in the client. LIBERO consumes the
  first seven dimensions, with its standard binary gripper conversion
  immediately before `env.step`.

`EMBODIMENT_ID` is required. Confirm it and the checkpoint's action-slot
semantics for LIBERO before treating success rates as valid. The published
model metadata does not provide a reliable public robot-to-embodiment mapping.
