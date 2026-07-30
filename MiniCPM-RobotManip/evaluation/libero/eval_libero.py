# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

from __future__ import annotations

import dataclasses
import json
import logging
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
LIBERO_PROMPT_TEMPLATE = (
    "The robot is LIBERO Franka, a simulated single-arm Franka manipulator. "
    "Its action control method is absolute single-arm end-effector pose in the "
    "unified 80D layout with gripper closed command, and its action FPS is "
    "20 Hz. Task: {instruction}"
)


def _matrix_to_rotation6d(matrix: object) -> np.ndarray:
    """Return LIBERO's contiguous-column rotation6D representation."""
    rotation = np.asarray(matrix, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix; got {rotation.shape}")
    return np.concatenate((rotation[:, 0], rotation[:, 1]))


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 20000

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
        max_steps = 800  # Longest training demo has 193 steps.
    elif args.task_suite_name == "libero_object":
        max_steps = 800  # Longest training demo has 254 steps.
    elif args.task_suite_name == "libero_goal":
        max_steps = 800  # Longest training demo has 270 steps.
    elif args.task_suite_name == "libero_10":
        max_steps = 800  # Longest training demo has 505 steps.
    elif args.task_suite_name == "libero_90":
        max_steps = 800  # Longest training demo has 373 steps.
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unified_ee6d=True,
    )

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
            model_instruction = LIBERO_PROMPT_TEMPLATE.format(
                instruction=task_description
            )

            # Reset environment.
            client_model.reset(task_description=model_instruction)
            env.reset()

            # Set initial states.
            obs = env.set_init_state(initial_states[episode_idx])
            controller = env.env.robots[0].controller
            controller.use_delta = True
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            controller.use_delta = False

            # Setup.
            t = 0
            replay_images = []
            full_actions = []

            logging.info("Starting episode %s...", task_episodes + 1)
            step = 0

            while t < max_steps:
                # Training uses a rotated agent view and an unmodified wrist view.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

                # Save the preprocessed agent view for replay video.
                replay_images.append(img)

                state = np.concatenate(
                    (
                        np.asarray(controller.ee_pos, dtype=np.float32).reshape(3),
                        _matrix_to_rotation6d(controller.ee_ori_mat),
                        np.zeros(1, dtype=np.float32),
                    )
                ).astype(np.float32)

                observation = {
                    "observation.primary": np.expand_dims(img, axis=0),
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # Keep the source camera order: [agentview, eye_in_hand].
                example_dict = {
                    "image": [
                        observation["observation.primary"][0],
                        observation["observation.wrist_image"][0],
                    ],
                    "lang": model_instruction,
                    "state": observation["observation.state"][0],
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

                if not (
                    world_vector_delta.size == 3
                    and rotation_delta.size == 3
                    and open_gripper.size == 1
                ):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, "
                        f"gripper={open_gripper.shape}"
                    )

                absolute_action = np.concatenate(
                    [world_vector_delta, rotation_delta, open_gripper], axis=0
                )
                full_actions.append(absolute_action)

                # Absolute OSC target: [xyz, axis-angle, binary gripper].
                obs, reward, done, info = env.step(absolute_action.tolist())
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
        env.close()

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
