# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Run the standard serial CALVIN long-horizon evaluation.

This is adapted from starVLA commit
631aae02afe6d95876e923ff518e8ff2ab9a2f88 and follows RoboFlamingo's
evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

The evaluator is an environment-side WebSocket client. The checkpoint is
loaded only by ``deployment.model_server.server_policy``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Set headless-rendering defaults before importing CALVIN. Existing environment
# values still win, so EGL or another backend can be selected by the caller.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import hydra
import numpy as np
import tyro
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from evaluation.libero.model2libero_interface import ModelClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360
DEFAULT_EVAL_SEQUENCES_PATH = Path(__file__).resolve().with_name(
    "eval_sequences.json"
)


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


@dataclasses.dataclass
class Args:
    """Command-line arguments exposed by tyro under the ``--args.*`` prefix."""

    # Policy server connection. The evaluator never loads a checkpoint.
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: int = 448

    # CALVIN paths. Explicit CLI values take precedence over environment values.
    calvin_root: Path | None = dataclasses.field(
        default_factory=lambda: _path_from_env("CALVIN_ROOT")
    )
    dataset_path: Path | None = dataclasses.field(
        default_factory=lambda: _path_from_env("CALVIN_DATASET_PATH")
    )
    calvin_config_path: Path | None = dataclasses.field(
        default_factory=lambda: _path_from_env("CALVIN_CONFIG_PATH")
    )
    eval_sequences_path: Path = DEFAULT_EVAL_SEQUENCES_PATH
    lang_annotation_cache: Path | None = dataclasses.field(
        default_factory=lambda: _path_from_env("LANG_ANNOTATION_CACHE")
    )

    # Evaluation settings.
    num_sequences: int = 1000
    seed: int = 0
    debug: bool = False
    eval_log_dir: Path = Path("tmp/calvin/eval_logs")
    reset: bool = False
    diverse_inst: bool = False


@dataclasses.dataclass(frozen=True)
class CalvinPaths:
    dataset: Path
    config: Path
    eval_sequences: Path
    lang_annotation_cache: Path | None


def resolve_calvin_paths(args: Args) -> CalvinPaths:
    """Resolve portable CALVIN paths without repository-specific constants."""

    calvin_root = args.calvin_root
    dataset_path = args.dataset_path
    config_path = args.calvin_config_path
    lang_annotation_cache = args.lang_annotation_cache

    if dataset_path is None and calvin_root is not None:
        dataset_path = calvin_root / "dataset" / "task_D_D"
    if config_path is None and calvin_root is not None:
        config_path = calvin_root / "calvin_models" / "conf"
    if lang_annotation_cache is None and calvin_root is not None:
        lang_annotation_cache = calvin_root / "lang_annotation_cache.json"

    if dataset_path is None:
        raise ValueError(
            "CALVIN dataset path is required. Set --args.dataset-path, "
            "CALVIN_DATASET_PATH, or --args.calvin-root/CALVIN_ROOT."
        )
    if config_path is None:
        raise ValueError(
            "CALVIN config path is required. Set --args.calvin-config-path, "
            "CALVIN_CONFIG_PATH, or --args.calvin-root/CALVIN_ROOT."
        )

    return CalvinPaths(
        dataset=dataset_path.expanduser().resolve(),
        config=config_path.expanduser().resolve(),
        eval_sequences=args.eval_sequences_path.expanduser().resolve(),
        lang_annotation_cache=(
            lang_annotation_cache.expanduser().resolve()
            if lang_annotation_cache is not None
            else None
        ),
    )


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Match starVLA's wire-size conversion without changing image geometry."""

    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


class CalvinPolicyClient:
    """CALVIN-specific observation and action adapter for ``ModelClient``."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 448,
    ) -> None:
        # ModelClient is transport-only: checkpoint loading and action scaling
        # belong to the server. MiniCPM actions require no unnormalization.
        self.client = ModelClient(
            host=host,
            port=port,
            image_size=[resize_size, resize_size],
        )
        self.step_count = 0
        # CALVIN's collect_plan helper checks this optional model attribute.
        self.plan = None

    def reset(self) -> None:
        """Reset action-chunk scheduling at each CALVIN subtask."""

        self.step_count = 0

    def step(self, obs: dict[str, Any], lang_annotation: str) -> np.ndarray:
        """Return one 7-D CALVIN action from the two synchronized RGB views."""

        rgb_static = _to_uint8(obs["rgb_obs"]["rgb_static"])
        rgb_gripper = _to_uint8(obs["rgb_obs"]["rgb_gripper"])
        example = {
            "image": [rgb_static, rgb_gripper],
            "lang": lang_annotation,
        }

        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(
            raw_action["world_vector"], dtype=np.float32
        ).reshape(-1)
        rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float32
        ).reshape(-1)
        open_gripper = np.asarray(
            raw_action["open_gripper"], dtype=np.float32
        ).reshape(-1)
        action = np.concatenate(
            [world_vector, rotation_delta, open_gripper], axis=0
        ).astype(np.float32)
        if action.shape != (7,):
            raise ValueError(
                "CALVIN requires a 7-D action "
                f"(3 translation + 3 rotation + 1 gripper), got {action.shape}"
            )

        self.step_count += 1
        return action


def make_env(dataset_path: Path):
    """Initialize CALVIN without the tactile camera used by pyrender."""

    val_folder = dataset_path / "validation"
    config_path = val_folder / ".hydra" / "merged_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"CALVIN validation config not found: {config_path}. "
            "dataset_path must directly contain validation/."
        )

    cfg = OmegaConf.load(config_path)
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        cfg.env.cameras = OmegaConf.create(
            {
                key: value
                for key, value in cfg.env.cameras.items()
                if key != "tactile"
            }
        )

    return hydra.utils.instantiate(
        cfg.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )


def evaluate_policy(
    policy: CalvinPolicyClient,
    env: Any,
    epoch: int,
    calvin_conf_path: Path,
    eval_sequences_path: Path,
    num_sequences: int,
    eval_log_dir: Path,
    lang_annotation_cache: Path | None = None,
    debug: bool = False,
    reset: bool = False,
    diverse_inst: bool = False,
) -> list[int]:
    """Evaluate the first ``num_sequences`` standard CALVIN task chains."""

    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")

    task_config = (
        calvin_conf_path / "callbacks" / "rollout" / "tasks"
        / "new_playtable_tasks.yaml"
    )
    annotation_config = (
        calvin_conf_path / "annotations" / "new_playtable_validation.yaml"
    )
    if not task_config.is_file():
        raise FileNotFoundError(f"CALVIN task config not found: {task_config}")

    task_oracle = hydra.utils.instantiate(OmegaConf.load(task_config))
    if diverse_inst:
        if lang_annotation_cache is None:
            raise ValueError(
                "diverse_inst requires --args.lang-annotation-cache, "
                "LANG_ANNOTATION_CACHE, or CALVIN_ROOT/lang_annotation_cache.json"
            )
        if not lang_annotation_cache.is_file():
            raise FileNotFoundError(
                f"Language annotation cache not found: {lang_annotation_cache}"
            )
        with lang_annotation_cache.open("r", encoding="utf-8") as file:
            val_annotations = json.load(file)
    else:
        if not annotation_config.is_file():
            raise FileNotFoundError(
                f"CALVIN validation annotations not found: {annotation_config}"
            )
        val_annotations = OmegaConf.load(annotation_config)

    requested_log_dir = eval_log_dir.expanduser().resolve()
    requested_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_log_dir = get_log_dir(str(requested_log_dir))

    if not eval_sequences_path.is_file():
        raise FileNotFoundError(
            f"CALVIN evaluation sequences not found: {eval_sequences_path}"
        )
    with eval_sequences_path.open("r", encoding="utf-8") as file:
        all_eval_sequences = json.load(file)
    eval_sequences = all_eval_sequences[:num_sequences]
    logger.info(
        "Evaluating %d of %d CALVIN sequences; outputs: %s",
        len(eval_sequences),
        len(all_eval_sequences),
        resolved_log_dir,
    )

    results: list[int] = []
    plans: defaultdict[str, list[Any]] = defaultdict(list)
    sequence_iterator = (
        eval_sequences
        if debug
        else tqdm(eval_sequences, position=0, leave=True)
    )

    for sequence_i, (initial_state, eval_sequence) in enumerate(
        sequence_iterator
    ):
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            resolved_log_dir,
            sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            sequence_iterator.set_description(
                " ".join(
                    [
                        f"{index + 1}/5 : {value * 100:.1f}% |"
                        for index, value in enumerate(count_success(results))
                    ]
                )
                + "|"
            )

    print_and_save(
        results,
        eval_sequences,
        resolved_log_dir,
        epoch,
    )
    return results


def evaluate_sequence(
    env: Any,
    policy: CalvinPolicyClient,
    task_checker: Any,
    initial_state: dict[str, Any],
    eval_sequence: list[str],
    val_annotations: Any,
    plans: defaultdict[str, list[Any]],
    debug: bool,
    eval_log_dir: str | Path = "",
    sequence_i: int = -1,
    reset: bool = False,
    diverse_inst: bool = False,
) -> int:
    """Evaluate one five-instruction CALVIN sequence."""

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print("\n\nEvaluating sequence: " + " -> ".join(eval_sequence))
        print("Subtask: ", end="")

    for subtask_i, subtask in enumerate(eval_sequence):
        reset_kwargs = (
            {"robot_obs": robot_obs, "scene_obs": scene_obs} if reset else {}
        )
        success = rollout(
            env,
            policy,
            task_checker,
            subtask,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            subtask_i,
            sequence_i,
            diverse_inst=diverse_inst,
            **reset_kwargs,
        )
        if not success:
            return success_counter
        success_counter += 1

    return success_counter


def rollout(
    env: Any,
    policy: CalvinPolicyClient,
    task_oracle: Any,
    subtask: str,
    val_annotations: Any,
    plans: defaultdict[str, list[Any]],
    debug: bool,
    eval_log_dir: str | Path = "",
    subtask_i: int = -1,
    sequence_i: int = -1,
    robot_obs: np.ndarray | None = None,
    scene_obs: np.ndarray | None = None,
    diverse_inst: bool = False,
) -> bool:
    """Roll out one language-conditioned subtask for at most ``EP_LEN`` steps."""

    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    obs = env.get_obs()
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation = lang_annotation.replace("\u2019", "'")

    policy.reset()
    start_info = env.get_info()
    if debug:
        img_queue = []

    for step in range(EP_LEN):
        action = policy.step(obs, lang_annotation)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, _, current_info = env.step(action)
        if debug:
            img_queue.append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
        if step == 0:
            collect_plan(policy, plans, subtask)

        current_task_info = task_oracle.get_task_info_for_set(
            start_info,
            current_info,
            {subtask},
        )
        if current_task_info:
            if debug:
                print(colored("success", "green"), end=" ")
                image_clip = ImageSequenceClip(img_queue, fps=30)
                image_clip.write_gif(
                    str(
                        Path(eval_log_dir)
                        / f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"
                    ),
                    fps=30,
                )
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
        image_clip = ImageSequenceClip(img_queue, fps=30)
        image_clip.write_gif(
            str(
                Path(eval_log_dir)
                / f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"
            ),
            fps=30,
        )
    return False


def main(args: Args) -> None:
    paths = resolve_calvin_paths(args)
    np.random.seed(args.seed)

    policy = CalvinPolicyClient(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
    )
    env = make_env(paths.dataset)
    evaluate_policy(
        policy=policy,
        env=env,
        epoch=0,
        calvin_conf_path=paths.config,
        eval_sequences_path=paths.eval_sequences,
        num_sequences=args.num_sequences,
        eval_log_dir=args.eval_log_dir,
        lang_annotation_cache=paths.lang_annotation_cache,
        debug=args.debug,
        reset=args.reset,
        diverse_inst=args.diverse_inst,
    )


if __name__ == "__main__":
    tyro.cli(main)
