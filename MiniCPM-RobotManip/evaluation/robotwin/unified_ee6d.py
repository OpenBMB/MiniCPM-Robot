# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Unified-80D RoboTwin layouts and controller conversions.

This module is the single source of truth for the RoboTwin evaluation data
contract. It mirrors the tensor layout used by the training artifacts without
importing the training repository, and it deliberately avoids SciPy so the
lightweight client can run inside the external RoboTwin environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MODEL_DIM = 80
LEFT_JOINT_START = 0
LEFT_JOINT_END = 6
LEFT_EEF_START = 7
LEFT_EEF_END = 17
LEFT_EEF_DIM = LEFT_EEF_END - LEFT_EEF_START
RIGHT_JOINT_START = 17
RIGHT_JOINT_END = 23
RIGHT_EEF_START = 24
RIGHT_EEF_END = 34
RIGHT_EEF_DIM = RIGHT_EEF_END - RIGHT_EEF_START
ROBOTWIN_EEF_DIM = LEFT_EEF_DIM + RIGHT_EEF_DIM
ROBOTWIN_JOINT_STATE_DIM = 14
ROBOTWIN_ENDPOSE_DIM = 16
ROBOTWIN_EE_ACTION_DIM = 16
GRIPPER_INDEX = 16
RIGHT_GRIPPER_INDEX = 33
GRIPPER_THRESHOLD = 0.5
ROBOTWIN_RAW_GRIPPER_OPEN_THRESHOLD = 0.4
_EPS = 1e-8


def robotwin_wxyz_to_rotate6d(quaternion: Any) -> np.ndarray:
    """Convert a RoboTwin ``wxyz`` quaternion to its interleaved rotation6D.

    RoboTwin training uses ``matrix[:, :2].reshape(6)``, i.e. the interleaved
    layout ``[r00, r01, r10, r11, r20, r21]``.

    The matrix is built in float64 with SciPy's ``as_matrix`` term grouping and
    only narrowed at the end, so the result is bit-identical to the SciPy-based
    reference implementation rather than merely equivalent.
    """

    wxyz = np.asarray(quaternion, dtype=np.float32).reshape(4).astype(np.float64)
    norm = np.linalg.norm(wxyz)
    if norm < _EPS:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        wxyz = wxyz / norm
    w, x, y, z = wxyz

    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    matrix = np.array(
        [
            [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ],
        dtype=np.float64,
    )
    return matrix[:, :2].reshape(6).astype(np.float32)


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to ``wxyz``, using SciPy's ``from_matrix`` branch order.

    SciPy picks the branch from a decision vector rather than from the trace
    sign, which fixes the quaternion sign it returns. Reproducing that choice
    keeps this client bit-comparable with the reference implementation
    instead of only equal up to the ``q``/``-q`` ambiguity.
    """

    diagonal = np.diag(matrix)
    decision = np.append(diagonal, diagonal.sum())
    choice = int(np.argmax(decision))
    xyzw = np.empty(4, dtype=np.float64)
    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3
        xyzw[i] = 1.0 - decision[3] + 2.0 * matrix[i, i]
        xyzw[j] = matrix[j, i] + matrix[i, j]
        xyzw[k] = matrix[k, i] + matrix[i, k]
        xyzw[3] = matrix[k, j] - matrix[j, k]
    else:
        xyzw[0] = matrix[2, 1] - matrix[1, 2]
        xyzw[1] = matrix[0, 2] - matrix[2, 0]
        xyzw[2] = matrix[1, 0] - matrix[0, 1]
        xyzw[3] = 1.0 + decision[3]
    xyzw = xyzw / np.linalg.norm(xyzw)
    return np.asarray(
        [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32
    )


def robotwin_rotate6d_to_wxyz(values: Any) -> np.ndarray:
    """Convert RoboTwin's interleaved rotation6D back to a ``wxyz`` quaternion."""

    rotation6d = np.asarray(values, dtype=np.float64).reshape(6)
    first = rotation6d[0::2]
    second = rotation6d[1::2]
    first_norm = np.linalg.norm(first)
    if first_norm < _EPS:
        raise ValueError("rotation6D first column is degenerate")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < _EPS:
        raise ValueError("rotation6D second column is degenerate")
    second = second / second_norm
    matrix = np.stack((first, second, np.cross(first, second)), axis=1)
    return _matrix_to_wxyz(matrix)


def robotwin_endpose_to_ee6d(values: Any) -> np.ndarray:
    """Convert RoboTwin's 16D endpose to dual-arm 20D EE6D physical values.

    Input per arm is ``xyz + quat_wxyz + gripper_open``. Output per arm is
    ``xyz + rot6d + gripper_closed``.
    """

    endpose = np.asarray(values, dtype=np.float32).reshape(-1)
    if endpose.size != ROBOTWIN_ENDPOSE_DIM:
        raise ValueError(f"Expected 16D RoboTwin endpose, got shape {endpose.shape}")
    # Compare in float32, matching the training/reference-client array comparison.
    # float64 promotion would flip the flag at a raw value of exactly 0.4.
    open_threshold = np.float32(ROBOTWIN_RAW_GRIPPER_OPEN_THRESHOLD)
    left_closed = 1.0 - float(endpose[7] > open_threshold)
    right_closed = 1.0 - float(endpose[15] > open_threshold)
    return np.concatenate(
        (
            endpose[0:3],
            robotwin_wxyz_to_rotate6d(endpose[3:7]),
            np.asarray([left_closed], dtype=np.float32),
            endpose[8:11],
            robotwin_wxyz_to_rotate6d(endpose[11:15]),
            np.asarray([right_closed], dtype=np.float32),
        )
    ).astype(np.float32)


def pack_robotwin_state(
    joint_state: Any,
    endpose: Any,
    *,
    ee6d_override: Any | None = None,
) -> np.ndarray:
    """Build the 80D RoboTwin state used by unified-80D checkpoints.

    ``joint_state`` follows the simulator/dataset order ``left_joint6,
    left_gripper, right_joint6, right_gripper``. The seventh joint slot of each
    six-DOF arm and channels 34:80 stay zero. EEF channels may be chained from
    the last commanded target through ``ee6d_override`` while joint channels
    remain measured.
    """

    joints = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    if joints.size != ROBOTWIN_JOINT_STATE_DIM:
        raise ValueError(f"Expected 14D RoboTwin joint state, got shape {joints.shape}")
    if ee6d_override is None:
        ee6d = robotwin_endpose_to_ee6d(endpose)
    else:
        ee6d = np.asarray(ee6d_override, dtype=np.float32).reshape(-1)
    if ee6d.size != ROBOTWIN_EEF_DIM:
        raise ValueError(f"Expected 20D RoboTwin EE6D state, got shape {ee6d.shape}")

    packed = np.zeros(MODEL_DIM, dtype=np.float32)
    packed[LEFT_JOINT_START:LEFT_JOINT_END] = joints[0:6]
    packed[LEFT_EEF_START:LEFT_EEF_END] = ee6d[0:10]
    packed[RIGHT_JOINT_START:RIGHT_JOINT_END] = joints[7:13]
    packed[RIGHT_EEF_START:RIGHT_EEF_END] = ee6d[10:20]
    return packed


def unpack_robotwin_eef(values: Any) -> np.ndarray:
    """Extract the dual-arm EE6D action channels from a unified-80D tensor."""

    packed = np.asarray(values, dtype=np.float32).reshape(-1)
    if packed.size != MODEL_DIM:
        raise ValueError(f"Expected {MODEL_DIM}D unified values, got shape {packed.shape}")
    return np.concatenate(
        (
            packed[LEFT_EEF_START:LEFT_EEF_END],
            packed[RIGHT_EEF_START:RIGHT_EEF_END],
        )
    ).astype(np.float32)


def robotwin_ee6d_to_env_action(
    values: Any,
    *,
    threshold: float = GRIPPER_THRESHOLD,
    gripper_close_position: float = 0.0,
) -> np.ndarray:
    """Convert dual-arm 20D EE6D values to RoboTwin's 16D absolute EE action.

    A channel is treated as closed when it is greater than *or equal to*
    ``threshold``. The reference client decides this as ``1.0 - value > 0.5``, which is
    bit-equivalent to ``value < 0.5`` for float32 inputs, so the boundary value
    itself must command a closed gripper.
    """

    ee6d = np.asarray(values, dtype=np.float32).reshape(-1)
    if ee6d.size != ROBOTWIN_EEF_DIM:
        raise ValueError(f"Expected 20D RoboTwin EE6D action, got shape {ee6d.shape}")
    closed_threshold = np.float32(threshold)
    left_gripper = gripper_close_position if ee6d[9] >= closed_threshold else 1.0
    right_gripper = gripper_close_position if ee6d[19] >= closed_threshold else 1.0
    decoded = np.concatenate(
        (
            ee6d[0:3],
            robotwin_rotate6d_to_wxyz(ee6d[3:9]),
            np.asarray([left_gripper], dtype=np.float32),
            ee6d[10:13],
            robotwin_rotate6d_to_wxyz(ee6d[13:19]),
            np.asarray([right_gripper], dtype=np.float32),
        )
    ).astype(np.float32)
    if decoded.shape != (ROBOTWIN_EE_ACTION_DIM,):
        raise ValueError(f"Unexpected RoboTwin EE action shape: {decoded.shape}")
    return decoded


def binarize_ee6d_grippers(
    values: Any, *, threshold: float = GRIPPER_THRESHOLD
) -> np.ndarray:
    """Return a copy of a 20D EE6D target with both gripper channels binarized.

    The training state at a chunk boundary follows the last absolute command,
    whose gripper channels are hard 0/1 rather than regressed values. The
    closed test matches :func:`robotwin_ee6d_to_env_action` and is inclusive of
    ``threshold``.
    """

    ee6d = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if ee6d.size != ROBOTWIN_EEF_DIM:
        raise ValueError(f"Expected 20D RoboTwin EE6D action, got shape {ee6d.shape}")
    closed_threshold = np.float32(threshold)
    ee6d[9] = 1.0 if ee6d[9] >= closed_threshold else 0.0
    ee6d[19] = 1.0 if ee6d[19] >= closed_threshold else 0.0
    return ee6d
