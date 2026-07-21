# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from evaluation.libero.model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # Resolution used to render training data.


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    value = float(arr[0])
    bin_val = 1.0 - 2.0 * (value > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Wait for objects to stabilize in simulation.
    num_trials_per_task: int = 50
    max_tasks: int = -1  # Positive values limit tasks for smoke tests.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "outputs/evaluation/libero"
    seed: int = 7


def eval_libero(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))

    # Set random seed.
    np.random.seed(args.seed)

    # Initialize LIBERO task suite.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # Longest training demo has 193 steps.
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # Longest training demo has 254 steps.
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # Longest training demo has 270 steps.
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # Longest training demo has 505 steps.
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # Longest training demo has 373 steps.
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(host=args.host, port=args.port)

    # Optional smoke-test cap; -1 evaluates the complete suite.
    n_eval_tasks = (
        num_tasks_in_suite
        if args.max_tasks <= 0
        else min(args.max_tasks, num_tasks_in_suite)
    )
    logging.info(
        "Evaluating %s of %s tasks (max_tasks=%s)",
        n_eval_tasks,
        num_tasks_in_suite,
        args.max_tasks,
    )

    # Start evaluation.
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        # Get task.
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states.
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description.
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed
        )

        # Start episodes.
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)

            # Reset environment.
            client_model.reset(task_description=task_description)
            env.reset()

            # Set initial states.
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup.
            t = 0
            replay_images = []
            full_actions = []

            logging.info("Starting episode %s...", task_episodes + 1)
            step = 0

            while t < max_steps + args.num_steps_wait:
                # Do nothing initially because the simulator drops objects and
                # they need time to settle.
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Rotate both views 180 degrees to match training preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save the preprocessed agent view for replay video.
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {
                    "observation.primary": np.expand_dims(img, axis=0),
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # Keep the source camera order: [agentview, eye_in_hand].
                # Standard MiniCPM LIBERO evaluation intentionally does not send
                # observation.state; the server supplies its zero80 behavior.
                example_dict = {
                    "image": [
                        observation["observation.primary"][0],
                        observation["observation.wrist_image"][0],
                    ],
                    "lang": observation["instruction"][0],
                }

                start_time = time.time()
                response = client_model.step(example=example_dict, step=step)
                end_time = time.time()
                del start_time, end_time

                raw_action = response["raw_action"]
                world_vector_delta = np.asarray(
                    raw_action.get("world_vector")
                ).reshape(-1)
                rotation_delta = np.asarray(
                    raw_action.get("rotation_delta")
                ).reshape(-1)
                open_gripper = np.asarray(
                    raw_action.get("open_gripper")
                ).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (
                    world_vector_delta.size == 3
                    and rotation_delta.size == 3
                    and open_gripper.size == 1
                ):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, "
                        f"gripper={gripper.shape}"
                    )

                delta_action = np.concatenate(
                    [world_vector_delta, rotation_delta, gripper], axis=0
                )
                full_actions.append(delta_action)

                # LIBERO consumes [xyz, axis-angle delta, binary gripper].
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video for every episode.
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)
            del full_actions

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info(
                "# successes: %s (%.1f%%)",
                total_successes,
                total_successes / total_episodes * 100,
            )

        logging.info(
            "Current task success rate: %s",
            float(task_successes) / float(task_episodes),
        )
        logging.info(
            "Current total success rate: %s",
            float(total_successes) / float(total_episodes),
        )

    logging.info(
        "Total success rate: %s",
        float(total_successes) / float(total_episodes),
    )
    logging.info("Total episodes: %s", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initialize the LIBERO environment and return its task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle, copied from robosuite.

    Source:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
