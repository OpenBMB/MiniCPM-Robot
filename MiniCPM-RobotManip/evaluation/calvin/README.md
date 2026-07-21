# CALVIN serial evaluation

This directory runs the standard 1,000-sequence CALVIN long-horizon evaluation
against the MiniCPM-RobotManip WebSocket policy server. It is a serial,
single-GPU path: there is no sequence sharding, multi-GPU launch, or result
aggregation.

The rollout loop and `eval_sequences.json` come from starVLA commit
`631aae02afe6d95876e923ff518e8ff2ab9a2f88` under the MIT License. The sequence
file contains the same 1,000 entries as that baseline (source SHA-256:
`c99e041c255df3408ebf3bbb4feea8b407e17e60e7c07d172a6ea7ba6680e8ca`).
OpenBMB modifications adapt paths and the policy client for
MiniCPM-RobotManip.

## Prerequisites

CALVIN is not vendored here. Install the external CALVIN environment and
download the original evaluation dataset. `CALVIN_DATASET_PATH` must point to
the `task_D_D` directory that directly contains `validation/`.

Use two Python environments:

- `MINICPM_PYTHON`: the MiniCPM-RobotManip environment, used to load the
  checkpoint and host the policy server.
- `CALVIN_PYTHON`: the external CALVIN environment, used for simulation and
  evaluation.

Install the thin-client dependencies into the CALVIN environment if they are
not already present:

```bash
/path/to/calvin/python -m pip install -r \
  /path/to/MiniCPM-RobotManip/evaluation/calvin/requirements-client.txt
```

The requirements file intentionally does not install CALVIN itself.

## Run server and evaluation

The launcher resolves repository paths from its own location, so it can be
called from any working directory:

```bash
MINICPM_PYTHON=/path/to/minicpm/python \
CALVIN_PYTHON=/path/to/calvin/python \
CHECKPOINT=openbmb/MiniCPM-RobotManip \
EMBODIMENT_ID=0 \
CALVIN_ROOT=/path/to/CALVIN \
CALVIN_DATASET_PATH=/path/to/task_D_D \
OUTPUT_ROOT=/path/to/calvin-results \
bash /path/to/MiniCPM-RobotManip/evaluation/calvin/eval_calvin.sh
```

The default endpoint is `127.0.0.1:10093`. The launcher starts one server,
waits for both valid MiniCPM handshake metadata and a protocol ping, runs the
serial evaluator, and terminates the server on exit or interruption. Server
and evaluator logs, CALVIN result files, and optional debug GIFs are written
under a timestamped directory in `OUTPUT_ROOT`.

`eval_calvin.sh` is a thin entry point. Environment validation, subprocess
groups, readiness, signal cleanup, and the run manifest are implemented in
`launcher.py`; the shared WebSocket probe lives in
`evaluation/common/probe_server.py`.

Use `bash eval_calvin.sh --help` for all environment overrides. Common ones
include `NUM_SEQUENCES`, `RESIZE_SIZE`, `EVAL_LOG_DIR`, `DEBUG=1`, `RESET=1`,
and `DIVERSE_INST=1`. Diverse instructions additionally require
`LANG_ANNOTATION_CACHE`; otherwise the standard CALVIN validation annotations
under `CALVIN_ROOT/calvin_models/conf` are used.

`EMBODIMENT_ID` is required. Set it to the ID that the selected checkpoint
uses for CALVIN; the public checkpoint does not provide a reliable ID-to-robot
mapping.

## Run the server separately

To keep the two environments in separate terminals, start only the policy
server with:

```bash
MINICPM_PYTHON=/path/to/minicpm/python \
CHECKPOINT=openbmb/MiniCPM-RobotManip \
EMBODIMENT_ID=0 \
HOST=127.0.0.1 \
PORT=10093 \
bash /path/to/MiniCPM-RobotManip/evaluation/calvin/run_policy_server.sh
```

`run_policy_server.sh` invokes
`python -m deployment.model_server.server_policy`. The checkpoint is loaded
only by that server; the evaluator has no checkpoint argument or
evaluator-side checkpoint semantics.

## Evaluation contract

For every CALVIN step, the client sends exactly two synchronized views in this
order:

1. `rgb_static`
2. `rgb_gripper`

The shared `evaluation.libero.model2libero_interface.ModelClient` manages
action-chunk scheduling. The evaluator consumes the first seven action values
as three translation deltas, three rotation deltas, and one gripper value. The
gripper value is converted to CALVIN's `+1` open / `-1` close convention before
`env.step`.

MiniCPM actions are execution-ready. Neither the client nor server normalizes
or unnormalizes them, and no `unnorm_key=franka` is used.

## Direct client invocation

When a compatible server is already running, invoke the evaluator directly
from the CALVIN environment:

```bash
PYTHONPATH=/path/to/MiniCPM-RobotManip \
/path/to/calvin/python -m evaluation.calvin.eval_calvin \
  --args.host 127.0.0.1 \
  --args.port 10093 \
  --args.calvin-root /path/to/CALVIN \
  --args.dataset-path /path/to/task_D_D \
  --args.eval-log-dir /path/to/calvin-results \
  --args.num-sequences 1000
```

The bundled `eval_sequences.json` is selected relative to `eval_calvin.py`,
independent of the current working directory.
