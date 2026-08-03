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

"""In-process wrapper around one external RMBench (RoboTwin 2.0 / SAPIEN) task.

RMBench stays an external installation: point ``RMBENCH_PATH`` at a checkout of
https://github.com/RoboTwin-Platform/RMBench with its ModelScope assets in place.
This module mirrors the argument assembly of ``RMBench/script/eval_policy.py::main``
so object placement, cameras, embodiment, and the per-task step limit are identical
to the benchmark.

The one deliberate departure from the upstream eval loop is the executor. Upstream
``take_action(qpos)`` re-plans a fresh rest-to-rest trajectory per waypoint, which
does not match how the training data was collected (a continuous dense sweep) and
makes even the expert's own trajectory fail. ``take_action_dense`` reproduces the
collection executor instead; ``eval_rmbench.py`` drives rollouts through it.

The action is RMBench's dual-arm absolute joint target (14-dim = left arm 6 + left
gripper 1 + right arm 6 + right gripper 1). This module has zero torch dependency
(numpy + sapien + yaml only).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import yaml

from evaluation.rmbench._sapien_env import prepare_sapien_runtime

# Vulkan loader + EGL vendor fixes -- must run before the first ``import sapien``.
prepare_sapien_runtime()

# Head + two wrist cameras, in the order the memVLA recipe expects.
SIM_CAMERAS = ("head_camera", "left_camera", "right_camera")

# Short camera names transported to the policy (simulator name -> policy key).
CAMERA_NAME_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def resolve_rmbench_path(rmbench_path: str | None = None) -> str:
    """Resolve and validate the external RMBench checkout."""
    repo = rmbench_path or os.environ.get("RMBENCH_PATH")
    if not repo:
        raise RuntimeError(
            "RMBENCH_PATH is not set. Clone https://github.com/RoboTwin-Platform/RMBench "
            "and download its ModelScope assets, then export RMBENCH_PATH to that checkout."
        )
    if not os.path.isdir(repo) or not os.path.isdir(os.path.join(repo, "envs")):
        raise RuntimeError(f"RMBench checkout not found at '{repo}' (no envs/ directory).")
    if not os.path.isdir(os.path.join(repo, "assets")):
        raise RuntimeError(
            f"RMBench checkout at '{repo}' is missing assets/. Download the assets and the "
            "vendored curobo data from ModelScope into the checkout (see evaluation/rmbench/README.md)."
        )
    return os.path.abspath(repo)


def _setup_rmbench_path(repo_root: str) -> None:
    """Put the RMBench tree on sys.path so envs / curobo / description imports resolve."""
    for path in (
        repo_root,
        os.path.join(repo_root, "envs", "curobo", "src"),
        os.path.join(repo_root, "policy"),
        os.path.join(repo_root, "description", "utils"),
        os.path.join(repo_root, "script"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


class RMBenchSimEnv:
    """One RMBench task environment with a continuous dense executor."""

    def __init__(self, task_name: str, task_config: str, repo_root: str):
        self.task_name = task_name
        self.task_config = task_config
        self.repo_root = resolve_rmbench_path(repo_root)
        _setup_rmbench_path(self.repo_root)
        # RMBench's scripts assume cwd == repo root for their many relative paths.
        os.chdir(self.repo_root)

        import importlib

        from envs import CONFIGS_PATH  # pyright: ignore[reportMissingImports]
        from envs.utils.create_actor import UnStableError  # pyright: ignore[reportMissingImports]

        self._UnStableError = UnStableError
        self._CONFIGS_PATH = CONFIGS_PATH

        # ---- argument assembly (mirrors eval_policy.main) ----
        with open(os.path.join(self.repo_root, "task_config", f"{task_config}.yml"), encoding="utf-8") as handle:
            args = yaml.load(handle.read(), Loader=yaml.FullLoader)
        args["task_name"] = task_name
        args["task_config"] = task_config
        args["ckpt_setting"] = None
        args["eval_video_log"] = False

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as handle:
            embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as handle:
            camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)

        def embodiment_file(name):
            robot_file = embodiment_types[name]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment files")
            return robot_file

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        def embodiment_cfg(robot_file):
            with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as handle:
                return yaml.load(handle.read(), Loader=yaml.FullLoader)

        args["left_embodiment_config"] = embodiment_cfg(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_cfg(args["right_robot_file"])
        args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
        args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
        args["eval_mode"] = True  # REQUIRED: otherwise step_lim stays None
        args["render_freq"] = 0

        self.args = args
        envs_module = importlib.import_module(f"envs.{task_name}")
        self.TASK_ENV = getattr(envs_module, task_name)()

        from generate_episode_instructions import (  # pyright: ignore[reportMissingImports]
            generate_episode_descriptions,
        )

        self._gen_instructions = generate_episode_descriptions
        self._closed = True

    # ------------------------------------------------------------------ seeds
    def find_solvable_seed(self, start_seed: int, max_tries: int = 200):
        """Find the next expert-solvable seed >= start_seed (RMBench's expert_check).

        Object layout for a given seed is deterministic, so the policy rollout on the
        returned seed sees the same layout the expert solved. Returns
        ``(seed, episode_info)`` or ``None``.
        """
        seed = start_seed
        for _ in range(max_tries):
            try:
                self.TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
                episode_info = self.TASK_ENV.play_once()
                solved = bool(self.TASK_ENV.plan_success and self.TASK_ENV.check_success())
                self.TASK_ENV.close_env()
            except self._UnStableError:
                self._safe_close()
                seed += 1
                continue
            except Exception:  # noqa: BLE001
                self._safe_close()
                seed += 1
                continue
            if solved:
                return seed, episode_info
            seed += 1
        return None

    def make_instruction(self, episode_info: dict, instruction_type: str = "seen", test_num: int = 100) -> str:
        results = self._gen_instructions(self.task_name, [episode_info["info"]], test_num)
        return str(np.random.choice(results[0][instruction_type]))

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int, instruction: str | None = None) -> dict[str, Any]:
        self.TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
        if instruction is not None:
            self.TASK_ENV.set_instruction(instruction=instruction)
        self._closed = False
        return self.TASK_ENV.get_obs()

    def get_obs(self) -> dict[str, Any]:
        return self.TASK_ENV.get_obs()

    @property
    def instruction(self) -> str:
        return self.TASK_ENV.get_instruction()

    @property
    def step_lim(self) -> int:
        return int(self.TASK_ENV.step_lim)

    @property
    def take_action_cnt(self) -> int:
        return int(self.TASK_ENV.take_action_cnt)

    @property
    def success(self) -> bool:
        return bool(self.TASK_ENV.eval_success)

    @property
    def done(self) -> bool:
        return self.success or (self.take_action_cnt >= self.step_lim)

    def take_action_dense(self, action: np.ndarray, substeps: int = 15) -> bool:
        """Execute ONE native-cadence waypoint as a continuous dense control step.

        Mirrors RMBench's data-collection executor (``set_arm_joints`` with a
        finite-difference velocity + ``set_gripper`` linearly ramped across the
        sub-steps + ``scene.step()`` per physics step) rather than the eval-time
        ``take_action`` that re-plans a rest-to-rest trajectory per waypoint. The
        policy emits fine native-cadence absolute joint targets exactly as the
        dataset stored them, so the arm must flow continuously through them.

        ``check_success`` is polled every physics ``scene.step()``, matching what
        upstream ``take_action`` does. The memory tasks' ``check_success`` is
        stateful (e.g. ``observe_and_pickup`` raises the occluding wall on the 20th
        physics step), so per-physics-step polling keeps the timing identical to the
        benchmark -- do not collapse it to one check per waypoint.

        ``substeps`` is the physics sub-steps per native frame. The RMBench scene
        runs at 250 Hz and collection saves a frame every 15 steps, so one dataset
        frame == 15 physics steps (~16.7 Hz, not the 50 Hz the LeRobot converter
        hardcoded). ``substeps=15`` matches both the trained history cadence and the
        step-limit budget. Returns ``True`` if success latched during this waypoint.
        """
        TE = self.TASK_ENV
        if TE.take_action_cnt >= TE.step_lim or TE.eval_success:
            return bool(TE.eval_success)

        robot = TE.robot
        la = int(self.args["left_arm_dim"])
        ra = int(self.args["right_arm_dim"])
        q = np.asarray(action, dtype=np.float64)
        left_arm = q[:la]
        left_grip = float(q[la])
        right_arm = q[la + 1: la + 1 + ra]
        right_grip = float(q[la + 1 + ra])

        # Finite-difference joint velocity from the CURRENT commanded drive targets over the
        # native interval, so set_arm_joints tracks continuously rather than stopping at each
        # waypoint. get_*_arm_jointState returns joint drive targets (the same values get_obs
        # reports), so this is numerically identical to reading get_obs but skips a full render.
        cur = np.asarray(
            robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
            dtype=np.float64,
        )
        dt = float(TE.scene.get_timestep())
        denom = max(substeps, 1) * dt
        left_vel = (left_arm - cur[:la]) / denom
        right_vel = (right_arm - cur[la + 1: la + 1 + ra]) / denom

        # Linearly ramp the gripper across the sub-steps (start -> target), mirroring the
        # collection executor instead of slamming the target on every sub-step.
        n_sub = max(int(substeps), 1)
        left_grip_seq = np.linspace(float(cur[la]), left_grip, n_sub + 1)[1:]
        right_grip_seq = np.linspace(float(cur[la + 1 + ra]), right_grip, n_sub + 1)[1:]

        TE.take_action_cnt += 1
        latched = False
        for s in range(n_sub):
            robot.set_arm_joints(left_arm, left_vel, "left")
            robot.set_gripper(float(left_grip_seq[s]), "left")
            robot.set_arm_joints(right_arm, right_vel, "right")
            robot.set_gripper(float(right_grip_seq[s]), "right")
            TE.scene.step()
            if TE.check_success():
                TE.eval_success = True
                latched = True
                break
        TE._update_render()
        return bool(latched or TE.eval_success)

    def _safe_close(self):
        try:
            self.TASK_ENV.close_env()
        except Exception:  # noqa: BLE001
            pass

    def close(self, clear_cache: bool = False):
        if not self._closed:
            try:
                self.TASK_ENV.close_env(clear_cache=clear_cache)
            except Exception:  # noqa: BLE001
                self._safe_close()
            self._closed = True


def encode_obs(observation: dict) -> dict[str, np.ndarray]:
    """RoboTwin obs -> {policy camera key: HWC uint8 rgb} (head + two wrist cams)."""
    cams = observation["observation"]
    return {
        dst: np.asarray(cams[src]["rgb"], dtype=np.uint8)
        for src, dst in CAMERA_NAME_MAP.items()
    }


def obs_state(observation: dict) -> np.ndarray:
    """14-dim dual-arm joint state vector (proprio condition)."""
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
