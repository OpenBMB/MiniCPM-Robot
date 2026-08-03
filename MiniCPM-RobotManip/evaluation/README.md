# MiniCPM-RobotManip simulation evaluation

This package migrates the standard starVLA `data.actions` evaluation paths for
LIBERO, CALVIN, and RoboTwin. The simulator source code, datasets, and assets
remain external installations.

The migration is based on
[starVLA commit `631aae02`](https://github.com/starVLA/starVLA/tree/631aae02afe6d95876e923ff518e8ff2ab9a2f88).
Evaluator control flow and environment-specific action conversion are retained,
while model loading and wire transport use the MiniCPM-RobotManip deployment
server.

## Layout

- `libero/`: serial evaluation and bounded multi-GPU suite scheduling.
- `calvin/`: serial 1,000-sequence long-horizon evaluation.
- `robotwin/`: single-task evaluation and multi-GPU FIFO scheduling.
- `rmbench/`: history-conditioned closed-loop evaluation with per-GPU server/driver pairs.
- `common/`: source-compatible helpers shared by migrated evaluators.

Each benchmark README documents its external environment and commands:

- [LIBERO](libero/README.md)
- [CALVIN](calvin/README.md)
- [RoboTwin](robotwin/README.md)
- [RMBench](rmbench/README.md)

## Common policy contract

All evaluators connect to a WebSocket + MessagePack server under
`deployment/model_server`. LIBERO, CALVIN, and RoboTwin use `server_policy`: a
request contains exactly one current time step and one or more ordered camera
views. RMBench uses `server_policy_memvla`: a request carries a window of frames
per camera and the response adds the generated sub-task text. Both return
execution-ready `float32` actions with shape
`(1, action_chunk_size, action_dim)`; the published checkpoint uses
`(1, 30, 80)`.

Neither the server nor these evaluators normalize or unnormalize model
actions. Environment adapters only perform operations required by the target
API:

- LIBERO consumes unified channels `7:17` as an absolute EE6D target.
- CALVIN consumes unified channels `7:17`, converts rotation6D to Euler xyz,
  and maps the gripper to the simulator convention.
- RoboTwin packs measured EEF poses into unified channels `7:17` and `24:34`
  (joint channels are zero by default), then extracts those EE6D channels as
  absolute dual-arm end-effector targets.
- RMBench packs measured 14D joints into unified 80D and extracts the sparse
  absolute-joint channels back, running the checkpoint's history + sub-task path.

The simulator clients map robot state into the checkpoint's unified-80D
channel layout before sending it to MiniCPM.

RoboTwin2 ALOHA uses embodiment ID 4 for the unified checkpoint.

## Environments

Model server and simulator run in separate Python environments:

- `MINICPM_PYTHON`: MiniCPM model dependencies and GPU inference.
- `LIBERO_PYTHON`, `CALVIN_PYTHON`, `ROBOTWIN_PYTHON`, or `RMBENCH_PYTHON`:
  the corresponding simulator plus the benchmark's `requirements-client.txt`.

The lightweight client supports Python 3.8 simulator environments through
`websockets>=13.1,<14`; the MiniCPM model-server environment remains Python
3.10 with `websockets==16.0`.

Launchers derive `MiniCPM-RobotManip` from their own path, so they can be
called from any working directory. Outputs default to
`MiniCPM-RobotManip/outputs/evaluation/<benchmark>/`.

The LIBERO, CALVIN, and RoboTwin launchers record the checkpoint reference and
available local repository HEAD revisions in run manifests. RMBench instead
writes per-GPU results and a merged `summary.json`. A mutable Hub model ID isn't
an immutable weight revision; use a verified local snapshot when exact result
reproduction is required.

LIBERO and RoboTwin include multi-GPU schedulers. They use one model server
and one evaluator per GPU slot, unique ports, metadata/ping readiness checks,
and signal-safe process cleanup. RMBench uses a shell orchestrator that pairs one
server and one driver per GPU with a history/subtask readiness handshake. CALVIN
intentionally remains serial.

Real success-rate validation requires the external simulator, assets, a GPU
checkpoint, and confirmed embodiment/action semantics.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and
license notices.
