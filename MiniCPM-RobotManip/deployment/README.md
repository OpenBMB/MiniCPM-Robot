# MiniCPM-RobotManip WebSocket deployment

This server implements the WebSocket + MessagePack/NumPy contract used by
starVLA evaluators. The first release targets LIBERO, CALVIN, and RoboTwin.

## Start the server

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

The default host is local-only. Bind to `0.0.0.0` only on a trusted network or
behind an authenticated proxy.

## Inference contract

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
        "actions": actions,  # np.float32, shape (1, 30, 80)
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
  `[agentview, eye_in_hand]`; the evaluator consumes the first 7 action
  dimensions.
- **CALVIN:** one frame with views ordered as
  `[rgb_static, rgb_gripper]`; the evaluator consumes the first 7 dimensions.
- **RoboTwin:** one frame with views ordered as
  `[head, left_wrist, right_wrist]`; the client omits its incompatible 14-D
  state and consumes the first 14 dimensions with the required joint reorder.

The migrated evaluators live under `MiniCPM-RobotManip/evaluation`:

```bash
# LIBERO multi-GPU
LIBERO_HOME=/path/to/LIBERO GPU_LIST="0 1" EMBODIMENT_ID=0 \
bash evaluation/libero/auto_eval_scripts/auto_eval_libero.sh \
  --checkpoint openbmb/MiniCPM-RobotManip

# CALVIN serial
MINICPM_PYTHON=/path/to/minicpm/python \
CALVIN_PYTHON=/path/to/calvin/python \
CALVIN_ROOT=/path/to/CALVIN CALVIN_DATASET_PATH=/path/to/task_D_D \
EMBODIMENT_ID=0 \
bash evaluation/calvin/eval_calvin.sh

# RoboTwin multi-GPU
ROBOTWIN_PATH=/path/to/RoboTwin \
bash evaluation/robotwin/start_eval.sh \
  --mode demo_clean --run-name minicpm \
  --checkpoint openbmb/MiniCPM-RobotManip \
  --default-embodiment-id 0 all
```

All migrated clients resize to 448×448 and send no normalization, DDIM, or
evaluator-side checkpoint fields. The server process alone loads the model
checkpoint. See `evaluation/<benchmark>/README.md` for external simulator
setup, single-worker commands, and camera/action contracts.

LIBERO OpenPI, BEHAVIOR's `normalized_actions` response, and VLN-CE's text
generation protocol are different wire contracts and are not supported by
this server.

## Reserved streaming extension

Stateless `infer` and `predict_action` always mean one complete current frame.
They will remain unchanged when streaming is added.

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

## Smoke test

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

Run protocol and adapter tests without loading the checkpoint:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
