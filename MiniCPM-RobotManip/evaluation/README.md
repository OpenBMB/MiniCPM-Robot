# MiniCPM-RobotManip simulation evaluation

This package migrates the standard starVLA `data.actions` evaluation paths for
LIBERO, CALVIN, and RoboTwin. The simulator source code, datasets, and assets
remain external installations.

The migration is based on starVLA commit
`631aae02afe6d95876e923ff518e8ff2ab9a2f88`. Evaluator control flow and
environment-specific action conversion are retained, while model loading and
wire transport use the MiniCPM-RobotManip deployment server.

## Layout

- `libero/`: serial evaluation and bounded multi-GPU suite scheduling.
- `calvin/`: serial 1,000-sequence long-horizon evaluation.
- `robotwin/`: single-task evaluation and multi-GPU FIFO scheduling.
- `common/`: source-compatible helpers shared by migrated evaluators.
- `tests/`: protocol, adapter, and launcher tests that don't require simulators.

Each benchmark README documents its external environment and commands:

- [LIBERO](libero/README.md)
- [CALVIN](calvin/README.md)
- [RoboTwin](robotwin/README.md)

## Common policy contract

All evaluators connect to the WebSocket + MessagePack server under
`deployment/model_server`. A request contains exactly one current time step
and one or more ordered camera views. The server returns execution-ready
`float32` actions with shape `(1, 30, 80)`.

Neither the server nor these evaluators normalize or unnormalize model
actions. Environment adapters only perform operations required by the target
API:

- LIBERO and CALVIN consume the first 7 dimensions and convert the gripper to
  the simulator convention.
- RoboTwin applies the migrated 14-D joint/gripper reorder.

The simulator clients don't send their incompatible low-dimensional robot
state, so the server uses MiniCPM's existing 80-D zero-state behavior.

The public checkpoint doesn't document a reliable embodiment-ID mapping.
Always set the benchmark's `EMBODIMENT_ID` or
`--default-embodiment-id` to the value required by the selected checkpoint.

## Environments

Model server and simulator run in separate Python environments:

- `MINICPM_PYTHON`: MiniCPM model dependencies and GPU inference.
- `LIBERO_PYTHON`, `CALVIN_PYTHON`, or `ROBOTWIN_PYTHON`: the corresponding
  simulator plus the benchmark's `requirements-client.txt`.

The lightweight client supports Python 3.8 simulator environments through
`websockets>=13.1,<14`; the MiniCPM model-server environment remains Python
3.10 with `websockets==16.0`.

Launchers derive `MiniCPM-RobotManip` from their own path, so they can be
called from any working directory. Outputs default to
`MiniCPM-RobotManip/outputs/evaluation/<benchmark>/`.

Run manifests record the checkpoint reference and available local repository
HEAD revisions. A mutable Hub model ID isn't an immutable weight revision; use
a verified local snapshot when exact result reproduction is required.

LIBERO and RoboTwin include multi-GPU schedulers. They use one model server
and one evaluator per GPU slot, unique ports, metadata/ping readiness checks,
and signal-safe process cleanup. CALVIN intentionally remains serial.

## Tests

Tests that don't require simulator installations live under
`evaluation/tests`:

```bash
cd MiniCPM-RobotManip
python -m unittest discover -s evaluation/tests -p 'test_*.py' -v
python -m compileall -q evaluation
```

Run `bash -n` on launcher scripts before deployment. Real success-rate
validation still requires the external simulator, assets, a GPU checkpoint,
and confirmed embodiment/action semantics.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and
license notices.
