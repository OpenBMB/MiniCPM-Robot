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
  --default-embodiment-id 4 \
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
--gripper-threshold VALUE
--gripper-close-position VALUE
--dry-run
```

`--gripper-threshold` and `--gripper-close-position` are forwarded to every
evaluator and recorded in `run_manifest.tsv`. They take precedence over the
`ROBOTWIN_GRIPPER_CLOSED_THRESHOLD` / `ROBOTWIN_GRIPPER_CLOSE_POSITION`
environment variables, which in turn take precedence over the client defaults
(`0.5` and `0.0`). `eval.sh` accepts the same two flags for manual runs.

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

Options go before the positional arguments:

```bash
... eval.sh --gripper-threshold 0.45 adjust_bottle demo_clean manual_run 0 0
```

`eval.sh` creates a unique temporary deployment YAML, injects task, mode, run
name, seed, and dotted policy module through RoboTwin overrides, runs from the
external checkout, and removes the temporary file on exit.

## Policy contract

- Policy module:
  `evaluation.robotwin.model2robotwin_interface`
- Shared conversions: `evaluation.robotwin.unified_ee6d`
- Endpoint default: `127.0.0.1:10093`
- Camera order: `[head, left_wrist, right_wrist]`
- Image size: the client resizes each frame to `448 x 448` with OpenCV
  `INTER_AREA`, matching the reference evaluation client. The MiniCPM server then applies the
  checkpoint's PIL `448 x 448` resize, which is an identity resample at this
  size
- State: EEF and closed-gripper channels are `7:17` and `24:34`; joint and
  reserved channels remain zero. The EEF channels always carry the measured
  endpose from the current observation
- Action mode: absolute only; other modes fail before evaluation
- Request: no unnormalization key, DDIM, or normalization parameters
- Response: must be successful, finite, and shaped
  `(1, action_chunk_size, 80)`
- EE action: channels `7:17` and `24:34`; gripper-closed predictions greater
  than or equal to `ROBOTWIN_GRIPPER_CLOSED_THRESHOLD` (default `0.5`) become
  `ROBOTWIN_GRIPPER_CLOSE_POSITION` (default `0.0`), otherwise `1.0`
- Timing: each model target is submitted to RoboTwin once; no action repeat is
  applied. The chunk index is driven by the client's own policy-step counter,
  not by RoboTwin's `take_action_cnt`
- Embodiment: this release's checkpoint maps RoboTwin to ID `4`

The model prompt is:

```text
The robot is RoboTwin2 ALOHA-AgileX, a simulated dual-arm ALOHA-style manipulator. Its action control method is absolute dual-arm end-effector pose in the unified 80D layout with gripper closed commands, and its action FPS is 15 Hz. Task: {instruction}
```

Use `--default-embodiment-id 4` for the unified RoboTwin checkpoint. The
client also sends ID `4` explicitly with every request. The embodiment ID is
checkpoint metadata rather than part of this contract; a different unified-80D
checkpoint may use a different ID.

### Ablation switches

The EEF-feedback, resize, and prompt semantics are each behind an environment
flag so a single factor can be measured in isolation. The defaults below match
the reference RoboTwin evaluation client, so that the same observation
produces the same 16D EE command on both sides; the alternatives reproduce the
earlier MiniCPM-only behaviour. `ROBOTWIN_STATE_JOINTS` is off by default: the
joint channels stay zero, and turning it on is an experiment rather than part
of this contract.

```text
ROBOTWIN_CHAIN_EEF                  0     1 chains the last absolute command
ROBOTWIN_CLIENT_RESIZE              1     0 sends simulator-resolution frames
ROBOTWIN_PROMPT_STYLE               robotwin2   'legacy' uses the older wording
ROBOTWIN_STATE_JOINTS               0     1 fills joint channels 0:6 and 17:23
ROBOTWIN_GRIPPER_CLOSED_THRESHOLD   0.5
ROBOTWIN_GRIPPER_CLOSE_POSITION     0.0
ROBOTWIN_EVAL_STEP_LIMIT_BONUS      1000  0 uses the checkout's limits as-is
```

`ROBOTWIN_EVAL_STEP_LIMIT_BONUS` is added to *every* per-task evaluation step
limit, so the RoboTwin 2.0 defaults become `1400` for `adjust_bottle` and
`2700` for `put_bottles_dustbin`. RoboTwin reads `_eval_step_limit.yml` from
`envs._base_task.CONFIGS_PATH` lazily, so `get_model()` writes a bumped copy to
a temporary directory and repoints that attribute. The external checkout is
never modified and no launcher flag is required.

Action repeat, the launcher's shard/seed protocol, and simulator RNG seeding
are *not* aligned with the reference evaluation protocol yet, so success rates are still not
directly comparable with internal numbers.

The external RoboTwin checkout supplies the base evaluation step limits, which
this client then raises by `ROBOTWIN_EVAL_STEP_LIMIT_BONUS`. Attention backend
selection, checkpoint format conversion, and per-episode random sequences can
also differ from the original internal evaluation pipeline.

## License and provenance

The interface and launch structure were migrated from starVLA, copyright 2025
the starVLA community, under the MIT License. Modifications for
MiniCPM-RobotManip are copyright 2026 The OpenBMB Team.
