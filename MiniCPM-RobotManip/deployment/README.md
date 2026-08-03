# MiniCPM-RobotManip WebSocket deployment

This server implements the WebSocket + MessagePack/NumPy contract used by
the LIBERO, CALVIN, RoboTwin, and RMBench evaluators. LIBERO, CALVIN, and
RoboTwin use the single-frame policy server; RMBench uses a separate
history-conditioned memVLA entry point.

## Start the single-frame server

Run from `MiniCPM-RobotManip` so that both `vla_infer.py` and the `deployment`
package are importable:

```bash
conda activate MiniCPM-RobotManip
cd MiniCPM-RobotManip
python -m deployment.model_server.server_policy \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --device cuda \
  --host 127.0.0.1 \
  --port 10093 \
  --default-embodiment-id 0
```

Existing starVLA evaluators don't send `embodiment_id`. Select the ID required
by the target robot with `--default-embodiment-id`; the published checkpoint
doesn't provide a reliable public ID-to-robot mapping. A request-level
`embodiment_id` overrides the server default.

The default host is local-only. The server uses plaintext `ws://` and has no
built-in authentication or TLS. Bind to `0.0.0.0` only on a trusted network or
behind an authenticated TLS proxy. `--max-message-bytes` limits incoming frame
size, and `--idle-timeout` can close an unused server automatically.

## Single-frame inference contract

An existing starVLA client sends a flat MessagePack payload:

```python
{
    "examples": [
        {
            "image": [camera_0, camera_1],  # ordered uint8 HWC RGB arrays
            "lang": "Pick up the red block.",
            "state": state,                 # optional 80-D state
        }
    ],
    "unnorm_key": None,                     # accepted and ignored
    "do_sample": False,                     # accepted and ignored
    "use_ddim": True,                       # accepted and ignored
    "num_ddim_steps": 10,                   # accepted and ignored
    "cfg_scale": 1.0,                       # accepted and ignored
    "embodiment_id": 0,                     # optional
    "seed": 123,                            # optional
}
```

`examples` must contain exactly one current time step. Its `image` list may
contain multiple synchronized camera views, in training-time order. It must
not contain historical frames. `text` is accepted as an alias for `lang`.
Missing state is replaced by the model's existing 80-D zero-state behavior.
Every view is validated as RGB `uint8 HWC` and then resized by
`MiniCPMVLAInference` to 448×448.

The response matches current starVLA clients:

```python
{
    "status": "ok",
    "ok": True,
    "type": "inference_result",
    "request_id": "default",
    "data": {
        # np.float32, shape (1, action_chunk_size, action_dim);
        # (1, 30, 80) for the published checkpoint
        "actions": actions,
    },
}
```

MiniCPM-RobotManip actions are execution-ready. The server does not normalize,
un-normalize, clip, truncate, reorder, or otherwise transform action values.
It only adds the batch dimension required by starVLA. Consequently
`unnorm_key` is a compatibility-only no-op, and handshake metadata reports
`action_normalization="none"` and `available_unnorm_keys=[]`.

The server also accepts the versioned envelope:

```python
{
    "type": "infer",
    "request_id": "request-1",
    "payload": {
        "examples": [...],
    },
}
```

## Target evaluators

- **LIBERO:** one frame with views ordered as
  `[agentview, eye_in_hand]`; the evaluator consumes unified channels `7:17`
  as an absolute EE6D target.
- **CALVIN:** one frame with views ordered as
  `[rgb_static, rgb_gripper]`; the evaluator consumes unified channels `7:17`
  and converts them to CALVIN's native 7D absolute Cartesian action.
- **RoboTwin:** one frame with views ordered as
  `[head, left_wrist, right_wrist]`; the client packs measured EEF poses into
  unified channels `7:17` and `24:34` (joint channels are zero by default) and
  extracts those EE6D channels as absolute dual-arm end-effector targets.

The migrated evaluators live under `MiniCPM-RobotManip/evaluation`:

```bash
# LIBERO multi-GPU
MINICPM_PYTHON=/path/to/minicpm/python \
LIBERO_PYTHON=/path/to/libero/python \
LIBERO_HOME=/path/to/LIBERO GPU_LIST="0 1" EMBODIMENT_ID=0 \
bash evaluation/libero/auto_eval_scripts/auto_eval_libero.sh \
  --checkpoint openbmb/MiniCPM-RobotManip

# CALVIN serial
MINICPM_PYTHON=/path/to/minicpm/python \
CALVIN_PYTHON=/path/to/calvin/python \
CALVIN_ROOT=/path/to/CALVIN CALVIN_DATASET_PATH=/path/to/task_D_D \
EMBODIMENT_ID=1 \
bash evaluation/calvin/eval_calvin.sh

# RoboTwin multi-GPU
MINICPM_PYTHON=/path/to/minicpm/python \
ROBOTWIN_PYTHON=/path/to/robotwin/python \
ROBOTWIN_PATH=/path/to/RoboTwin \
bash evaluation/robotwin/start_eval.sh \
  --mode demo_clean --run-name minicpm \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --default-embodiment-id 4 all
```

All migrated clients resize to 448×448 and send no normalization, DDIM, or
evaluator-side checkpoint fields. The server process alone loads the model
checkpoint. See `evaluation/<benchmark>/README.md` for external simulator
setup, single-worker commands, and camera/action contracts.

LIBERO OpenPI, BEHAVIOR's `normalized_actions` response, and VLN-CE's text
generation protocol are different wire contracts and are not supported by
the single-frame server.

## RMBench history-conditioned server

RMBench uses the memVLA server rather than `server_policy`:

```bash
conda activate MiniCPM-RobotManip
cd MiniCPM-RobotManip
python -m deployment.model_server.server_policy_memvla \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --device cuda \
  --host 127.0.0.1 \
  --port 10094 \
  --default-embodiment-id 4
```

One request contains exactly one history window. `examples[0].views` is an
ordered list of per-camera frame lists for `[head, left_wrist, right_wrist]`.
The published recipe requires 60 strided head-camera frames plus the current
frame from each wrist camera (62 frames total). The response contains
`data.actions` with shape `(1, 30, 80)` and generated `data.subtask` text.
Handshake capabilities report `history=true` and `subtask=true`.

Use the benchmark launcher for normal evaluation; it starts one memVLA server
and one simulator driver per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RMBENCH_PATH=/path/to/RMBench \
MINICPM_PYTHON=/path/to/minicpm/python \
RMBENCH_PYTHON=/path/to/rmbench/python \
bash evaluation/rmbench/start_eval.sh \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --seeds-per-task 25
```

See `evaluation/rmbench/README.md` for the frozen history recipe, simulator
setup, rendering requirements, and action mapping.

## Reserved streaming extension

For the single-frame server, stateless `infer` and `predict_action` always mean
one complete current frame. They will remain unchanged when streaming is added.

Protocol version 1 reserves four envelope message types:

- `session.open`: create a model-context session and return `session_id`.
- `stream.infer`: send `session_id`, monotonic `sequence_id`, and one current
  frame.
- `session.reset`: clear the context for `session_id`.
- `session.close`: release `session_id`.

The current model adapter implements only `FramePolicy`, so handshake
capabilities report `streaming=false` and `sessions=false`. Reserved calls
return `capability_not_supported` without closing the connection.

`WebsocketPolicyServer.register_handler()` can replace each reserved route.
Future native-cache support can implement `StreamingPolicy` plus a separate
session manager and register those handlers without changing MessagePack
encoding, the receive loop, error responses, or stateless inference.
Multi-view images are never interpreted as temporal history.

## Single-frame smoke test

Start `deployment.model_server.server_policy` first, then run this from the
`MiniCPM-RobotManip` directory in an environment containing the deployment
client dependencies:

```python
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

image = np.zeros((448, 448, 3), dtype=np.uint8)
with WebsocketClientPolicy("127.0.0.1", 10093) as client:
    print(client.get_server_metadata())
    response = client.predict_action({
        "examples": [{"image": [image], "lang": "Move forward."}],
    })
    print(response["data"]["actions"].shape)
```
