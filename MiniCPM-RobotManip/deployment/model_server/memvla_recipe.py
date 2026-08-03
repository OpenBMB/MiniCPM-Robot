# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen memVLA recipe constants for the RMBench checkpoint.

Every value here mirrors the training configuration byte for byte. The prompt is
part of the model's input distribution: the checkpoint generates a sub-task from
it, and the action head is conditioned on the hidden states of that same
sequence. A drifting prompt does not raise -- it silently degrades closed-loop
behaviour. Treat this module as a checkpoint constant, not a tuning surface.

Standard-library only, so the simulator environment can import it alongside the
GPU model environment.
"""

from __future__ import annotations


# Dataset frame rate. The LeRobot ``info.json`` reports 50, which is mislabeled:
# the scene runs at 250 Hz and stores one frame every 15 physical steps.
NATIVE_FPS = 250.0 / 15.0
CONTROL_FREQUENCY_HZ = round(NATIVE_FPS)  # 17

ACTION_HORIZON = 30
UNIFIED_DIM = 80
EMBODIMENT_ID = 4

# Prompt camera labels, in the same order as the transported views.
CAMERA_LABELS = ("Head camera", "Left wrist camera", "Right wrist camera")

# hist60s: ~60 s of head history at ~1 FPS (17 frames is about 1.02 s), plus the
# current frame from each wrist.
HEAD_N_FRAMES, HEAD_STRIDE = 60, 17
WRIST_N_FRAMES, WRIST_STRIDE = 1, 1

# (n_frames, stride) per view, matching CAMERA_LABELS.
VIEW_SPECS = (
    (HEAD_N_FRAMES, HEAD_STRIDE),
    (WRIST_N_FRAMES, WRIST_STRIDE),
    (WRIST_N_FRAMES, WRIST_STRIDE),
)
NUM_VIEWS = len(VIEW_SPECS)
TOTAL_FRAMES = sum(n_frames for n_frames, _ in VIEW_SPECS)  # 62

IMAGE_SIZE = (448, 448)

# Unified 80D layout: left arm block 0-16, right arm block 17-33, reserved 34-79.
# Each arm block is joint(7) + xyz(3) + rot6d(6) + gripper(1). RMBench is
# absolute joint control, so only 6 real joints per arm are used; the 7th joint
# slot is padding and the xyz/rot6d segments stay zero and masked out of the loss.
LEFT_JOINT_SLICE = (0, 6)
RIGHT_JOINT_SLICE = (17, 23)
LEFT_GRIPPER_INDEX = 16
RIGHT_GRIPPER_INDEX = 33

# Simulator-side 14D layout: left joints, left gripper, right joints, right gripper.
SIM_ACTION_DIM = 14
SIM_LEFT_JOINT_SLICE = (0, 6)
SIM_LEFT_GRIPPER_INDEX = 6
SIM_RIGHT_JOINT_SLICE = (7, 13)
SIM_RIGHT_GRIPPER_INDEX = 13

# Gripper is trained in "closed" mode: the raw opening in [0, 1] (larger is more
# open) is binarised to 1 = closed. 0.4 thresholds the observed state, 0.5
# thresholds the predicted command.
GRIPPER_OPEN_THRESHOLD = 0.4
GRIPPER_CLOSED_THRESHOLD = 0.5

MAX_SUBTASK_TOKENS = 24

# "aligned" prompt style, matching the co-training run's ROBOT_TYPE_PROMPTS
# phrasing. The action FPS keeps RMBench's real 17 Hz rather than RoboTwin's 15.
EMBODIMENT_INTRO = (
    "The robot is RoboTwin ALOHA-AgileX, a dual-arm ALOHA-style manipulator. "
    "Its action control method is absolute dual-arm joint position in the unified 80D layout "
    f"with gripper closed command, and its action FPS is {CONTROL_FREQUENCY_HZ} Hz."
)


def hist_offsets(n_frames: int, stride: int) -> list[int]:
    """Stored-frame offsets for the most recent ``n_frames`` at ``stride``, oldest first."""
    return [-(n_frames - 1 - i) * stride for i in range(n_frames)]


def frame_ages_sec(offsets: list[int]) -> list[float]:
    """Age in seconds of each frame relative to the current one, at the real frame rate."""
    return [round(-offset / NATIVE_FPS, 2) for offset in offsets]


VIEW_FRAME_AGES_SEC = tuple(
    tuple(frame_ages_sec(hist_offsets(n_frames, stride))) for n_frames, stride in VIEW_SPECS
)


def frame_timestamp_tag(age_sec: float) -> str:
    """Per-frame timestamp text injected before each image."""
    return "[now]" if float(age_sec) == 0 else f"[t-{float(age_sec):.1f}s]"


def embodiment_prompt(instruction: str) -> str:
    """Build the trailing text of the user turn.

    Ported from ``PrepareMiniCPMMemVLAInputs._embodiment_prompt``. The instruction
    is normalised the same way training normalised it.
    """
    instruction = str(instruction).strip().replace("_", " ").rstrip(". ")
    if not instruction:
        raise ValueError("instruction must be a non-empty string")
    cameras_clause = (
        f"You are given the recent history video from {NUM_VIEWS} cameras "
        "(each frame is tagged with its timestamp in seconds). "
    )
    return (
        f"{EMBODIMENT_INTRO} "
        f"The next {ACTION_HORIZON} control actions are predicted from the current sub-task. "
        f"{cameras_clause}"
        f"The overall task is: {instruction}. "
        f"Identify the current sub-task."
    )


def validate_views(view_lengths: list[int]) -> None:
    """Check transported per-view frame counts against the frozen recipe."""
    expected = [n_frames for n_frames, _ in VIEW_SPECS]
    if list(view_lengths) != expected:
        raise ValueError(
            f"views must contain {expected} frames per camera "
            f"(order: {list(CAMERA_LABELS)}); got {list(view_lengths)}"
        )
