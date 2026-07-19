# -*- coding: utf-8 -*-
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

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .policy import MiniCPMRobotTrackPolicy
from .splits import split_episodes


TASK_CONFIGS = {
    "stt": "track_infer_stt.yaml",
    "at": "track_infer_at.yaml",
    "dt": "track_infer_dt.yaml",
}


# EVT-Bench's evaluation agents in TrackVLA and OmTrackVLA both submit this
# exact multi-action tuple on every simulator step.  Keep it fixed: changing
# the set based on episode metadata changes the benchmark dynamics.
BENCHMARK_ACTIONS = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scene_key(scene_id: str) -> str:
    name = Path(scene_id).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _write_episode_result(
    output_root: Path,
    episode: Any,
    result: Dict[str, Any],
    record_infos: list[Dict[str, Any]],
) -> None:
    directory = output_root / scene_key(episode.scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    info_path = directory / f"{episode.episode_id}_info.json"
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(record_infos, handle, indent=2)
    path = directory / f"{episode.episode_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def _set_lighting(simulator) -> None:
    from habitat_sim.gfx import LightInfo, LightPositionModel

    simulator.set_light_setup(
        [
            LightInfo(
                vector=vector,
                color=[1.0, 1.0, 1.0],
                model=LightPositionModel.Global,
            )
            for vector in (
                [10.0, -2.0, 0.0, 0.0],
                [-10.0, -2.0, 0.0, 0.0],
                [0.0, -2.0, 10.0, 0.0],
                [0.0, -2.0, -10.0, 0.0],
            )
        ]
    )


def _episode_action(velocity) -> Dict[str, Any]:
    return {
        "action": BENCHMARK_ACTIONS,
        "action_args": {"agent_1_base_vel": velocity},
    }


def evaluate_split(
    habitat,
    config,
    dataset,
    policy: MiniCPMRobotTrackPolicy,
    output_root: Path,
) -> None:
    with habitat.TrackEnv(config=config, dataset=dataset) as environment:
        for _ in range(len(environment.episodes)):
            environment.reset()
            _set_lighting(environment.sim)
            episode = environment.current_episode
            episode_info = getattr(episode, "info", {}) or {}
            instruction = episode_info.get("instruction")
            robot = environment.sim.agents_mgr[1].articulated_agent
            main_human = environment.sim.agents_mgr[0].articulated_agent

            step_count = 0
            followed_steps = 0
            too_far_steps = 0
            status = "Normal"
            record_infos: list[Dict[str, Any]] = []
            metrics: Dict[str, Any] = environment.get_metrics()
            while not environment.episode_over:
                observations = environment.sim.get_sensor_observations()
                environment.task._get_observations(episode)
                rgb = observations["agent_1_articulated_agent_jaw_rgb"]
                velocity = policy.act(rgb, instruction)
                step_count += 1
                environment.step(_episode_action(velocity))
                metrics = environment.get_metrics()

                if float(metrics.get("human_following", 0.0)) == 1.0:
                    followed_steps += 1
                    too_far_steps = 0

                distance_to_human = float(
                    np.linalg.norm(robot.base_pos - main_human.base_pos)
                )
                if distance_to_human > 4.0:
                    too_far_steps += 1
                    if too_far_steps > 20:
                        status = "Lost"
                        break

                record_infos.append(
                    {
                        "step": step_count,
                        "dis_to_human": distance_to_human,
                        "facing": metrics.get("human_following", 0.0),
                        "base_velocity": np.asarray(velocity).tolist(),
                    }
                )

                if float(metrics.get("human_collision", 0.0)) == 1.0:
                    status = "Collision"
                    break

            metrics = environment.get_metrics()
            policy.reset()
            final_following = bool(metrics.get("human_following", 0.0))
            if step_count < 300:
                success = bool(metrics.get("human_following_success", 0.0)) and final_following
            else:
                success = final_following
            result = {
                "finish": bool(environment.episode_over),
                "status": status,
                "success": success,
                "following_rate": followed_steps / step_count,
                "following_step": followed_steps,
                "total_step": step_count,
                "collision": float(metrics.get("human_collision", 0.0)),
            }
            if instruction is not None:
                result["instruction"] = instruction
            _write_episode_result(output_root, episode, result, record_infos)


def run(args: argparse.Namespace) -> Dict[str, object]:
    root = project_root()
    checkpoint = args.checkpoint.resolve()
    output_root = (
        args.output.resolve() if args.output is not None else root / "results" / args.task
    )
    vendor_path = str(root / "third_party" / "habitat-lab")
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.chdir(root)

    import habitat
    from habitat.datasets import make_dataset

    import evt_bench  # noqa: F401

    config_path = (
        root
        / "third_party"
        / "habitat-lab"
        / "habitat"
        / "config"
        / "benchmark"
        / "nav"
        / "track"
        / TASK_CONFIGS[args.task]
    )
    config = habitat.get_config(str(config_path))
    seed = int(config.habitat.simulator.seed)
    random.seed(seed)
    np.random.seed(seed)
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    split_dataset = copy.copy(dataset)
    split_dataset.episodes = split_episodes(
        dataset.episodes,
        args.split_count,
        args.split_id,
        seed,
    )
    policy = MiniCPMRobotTrackPolicy(
        checkpoint,
        torch.device(args.device),
        backbone_override=args.backbone,
    )
    evaluate_split(habitat, config, split_dataset, policy, output_root)
    result = {"split_id": args.split_id, "episodes": len(split_dataset.episodes)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniCPM-RobotTrack")
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Complete Hugging Face snapshot directory or a legacy training .pt file",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--split-count",
        type=int,
        default=30,
        help="Number of EVT-Bench dataset splits",
    )
    parser.add_argument("--split-id", type=int, required=True)
    parser.add_argument(
        "--backbone",
        default=None,
        help="Optional local MiniCPM path for legacy .pt checkpoints only",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.split_count <= 0:
        parser.error("--split-count must be positive")
    if args.split_id < 0 or args.split_id > args.split_count:
        parser.error("--split-id must be in [0, split-count], including remainder")
    if not args.checkpoint.exists() or not (
        args.checkpoint.is_file() or args.checkpoint.is_dir()
    ):
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
