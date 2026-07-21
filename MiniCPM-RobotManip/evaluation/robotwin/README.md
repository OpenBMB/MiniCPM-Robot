<!--
Copyright 2025 starVLA community. All rights reserved.
Licensed under the MIT License.
Modifications Copyright 2026 The OpenBMB Team.
-->

# MiniCPM-RobotManip RoboTwin evaluation

This directory adapts the standard RoboTwin policy interface and the starVLA
multi-GPU evaluation flow to MiniCPM-RobotManip. RoboTwin remains an external
installation; no RoboTwin source patch is required. The launcher uses
RoboTwin's generic `--overrides` support.

`start_eval.sh` is intentionally a thin entry point. CLI parsing, GPU slots,
ports, subprocess groups, readiness, cleanup, and manifests live in
`launcher.py`; `evaluation/common/probe_server.py` performs the WebSocket
metadata/ping check in the MiniCPM environment.

The migration is based on starVLA commit
`631aae02afe6d95876e923ff518e8ff2ab9a2f88`. See the license note below.

## Environments

Use two separate Python environments:

1. `MINICPM_PYTHON` loads the checkpoint and runs the WebSocket policy server.
2. `ROBOTWIN_PYTHON` runs the externally installed RoboTwin simulator and
   imports the lightweight policy client from this repository.

`LAUNCHER_PYTHON` runs the standard-library-only scheduler and defaults to
`python3`.

Install the client-only wire dependencies in the RoboTwin environment:

```bash
/path/to/robotwin/bin/python -m pip install \
  -r MiniCPM-RobotManip/evaluation/robotwin/requirements-client.txt
```

Set the runtime paths before evaluation:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export MINICPM_PYTHON=/path/to/minicpm/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin/bin/python
```

`ROBOTWIN_PATH/script/eval_policy.py` must exist and support the standard
`--config ... --overrides ...` interface.

## Multi-GPU evaluation

From any directory:

```bash
bash MiniCPM-RobotManip/evaluation/robotwin/start_eval.sh \
  --mode demo_clean \
  --run-name minicpm_clean \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --default-embodiment-id 0 \
  adjust_bottle
```

`--checkpoint` accepts either a Hugging Face Hub model ID or a local checkpoint
directory. A local checkpoint file is not supported by the MiniCPM loader.

Tasks can be supplied in four forms:

```bash
# Separate arguments
... adjust_bottle open_laptop

# Comma-separated
... adjust_bottle,open_laptop

# A file containing task names, comments, and/or comma-separated lines
... tasks.txt

# The built-in RoboTwin 2.0 list of 50 tasks
... all
```

The scheduler detects `CUDA_VISIBLE_DEVICES` first, then `nvidia-smi`. It builds
FIFO execution slots, allocates an available port to each slot starting at
`10093`, and starts a fresh server/evaluator pair for every task. The default is
one slot per GPU. More than one `--jobs-per-gpu` slot is allowed but warned
because each slot loads another model and simulator on the same GPU.

Useful options:

```text
--seed N
--jobs-per-gpu N
--base-port PORT
--server-timeout SECONDS
--default-embodiment-id ID
--dry-run
```

`--name` is an alias for `--run-name`. Run `start_eval.sh --help` for all
arguments and environment variables.

Before starting RoboTwin, readiness checks connect with this repository's
`WebsocketClientPolicy`, read server metadata, verify the checkpoint and
default embodiment ID, and issue a real protocol `ping`. The server PID is
checked throughout startup. Task failures are collected, and any failure makes
the launcher exit non-zero. SIGINT/SIGTERM recursively stop server, simulator,
and logging subprocesses.

Logs default to:

```text
MiniCPM-RobotManip/outputs/evaluation/robotwin/<run>_<mode>_<checkpoint>_<timestamp>_<pid>/
```

Set `OUTPUT_ROOT` to use another root. Every task has separate server and
evaluator logs; `run_manifest.tsv`, `schedule.tsv`, and `status.tsv` record the
run configuration, slot assignment, and final exit status.

## Manual single-task evaluation

Start the server in the MiniCPM environment:

```bash
MINICPM_PYTHON=/path/to/minicpm/bin/python \
bash MiniCPM-RobotManip/evaluation/robotwin/run_policy_server.sh \
  openbmb/MiniCPM-RobotManip 0 10093 127.0.0.1 cuda 0
```

Then run one task with the RoboTwin environment:

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/robotwin/bin/python \
bash MiniCPM-RobotManip/evaluation/robotwin/eval.sh \
  adjust_bottle demo_clean manual_run 0 0 10093 127.0.0.1
```

`eval.sh` creates a unique temporary deployment YAML, injects task, mode, run
name, seed, and dotted policy module through RoboTwin overrides, runs from the
external checkout, and removes the temporary file on exit.

## Policy contract

- Policy module:
  `evaluation.robotwin.model2robotwin_interface`
- Endpoint default: `127.0.0.1:10093`
- Camera order: `[head, left_wrist, right_wrist]`
- Client image size: `448 x 448`
- State: omitted from the request; RoboTwin's incompatible 14-dimensional
  joint vector is not forwarded, so the server supplies its zero80 default
- Action mode: absolute only; other modes fail before evaluation
- Request: no unnormalization key, DDIM, or normalization parameters
- Response: must be successful, finite, and shaped
  `(1, action_chunk_size, D)` with `D >= 14`
- RoboTwin action order:
  `[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]`

`--default-embodiment-id` is required because the published checkpoint does
not currently document a reliable embodiment-ID-to-robot mapping. Set it to
the value appropriate for the checkpoint and RoboTwin setup.

## License and provenance

The interface and launch structure were migrated from starVLA, copyright 2025
the starVLA community, under the MIT License. Modifications for
MiniCPM-RobotManip are copyright 2026 The OpenBMB Team.
