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

"""Closed-loop RMBench success-rate evaluation for the memVLA checkpoint.

One process, one SAPIEN environment, one WebSocket policy server. The launcher
runs one server + one driver per GPU and shards work across GPUs; this driver runs
its assigned tasks serially. For each seed it resets the task (optionally filtering
to expert-solvable seeds), then rolls out a receding-horizon loop: build the
strided history, ask the server for one 80D chunk, decode it to 14D joint targets,
execute ``execute_horizon`` of them through the continuous dense executor, and
observe every native frame back into the history buffer.

Held-out seed protocol: seeds start at ``base_seed=100000``. Demos were collected
on small seeds (0..N), so evaluation must use a disjoint held-out layout. By
default each seed is validated expert-solvable through RMBench's (slow) CuRobo
solve, which also generates the task language.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.rmbench.model2rmbench_interface import RMBenchPolicyClient  # noqa: E402
from evaluation.rmbench.rmbench_env import RMBenchSimEnv, encode_obs, obs_state  # noqa: E402

# Diagnostic switch: print the sub-task the server returns each decision. Off by
# default; adds only a print, so it never affects the evaluated result.
_LOG_SUBTASK = os.environ.get("RMBENCH_LOG_SUBTASK", "0") == "1"

# The 10 implemented RMBench tasks.
DEFAULT_TASKS = [
    "observe_and_pickup",
    "put_back_block",
    "swap_T",
    "press_button",
    "place_block_mat",
    "battery_try",
    "rearrange_blocks",
    "swap_blocks",
    "cover_blocks",
    "blocks_ranking_try",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks (default: all 10).")
    ap.add_argument("--task_config", default="demo_clean")
    ap.add_argument("--instruction_type", default="seen")
    ap.add_argument("--seeds_per_task", type=int, default=25)
    ap.add_argument("--execute_horizon", type=int, default=16,
                    help="Execute the first N of each predicted chunk, then re-decide.")
    ap.add_argument("--dense_substeps", type=int, default=15,
                    help="Physics sub-steps per native waypoint (one dataset frame = 15 steps).")
    ap.add_argument("--max_seed_search", type=int, default=50)
    ap.add_argument("--base_seed", type=int, default=100000,
                    help="Held-out eval seeds start here; demos were collected on small seeds.")
    ap.add_argument("--seed_pool_offset", type=int, default=0,
                    help="Start index into the seed schedule. For a seed-level multi-GPU split, "
                         "lane i passes offset=i*share so lanes draw disjoint seeds.")
    ap.add_argument("--expert_check", action="store_true", default=True,
                    help="Validate each seed is expert-solvable + generate task language.")
    ap.add_argument("--no_expert_check", dest="expert_check", action="store_false",
                    help="Skip the (slow) CuRobo expert solve / seed filtering.")
    ap.add_argument("--rmbench_path", default=None, help="External RMBench checkout (or $RMBENCH_PATH).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10094)
    ap.add_argument("--response_timeout", type=float, default=300.0)
    ap.add_argument("--video_dir", default=None,
                    help="If set, record one success + one failure MP4 per task under <video_dir>/<task>/.")
    ap.add_argument("--video_max_per_task", type=int, default=1)
    ap.add_argument("--results_json", default=None, help="Where to write the structured result table.")
    return ap.parse_args()


def _resolve_seed(task_env: RMBenchSimEnv, args: argparse.Namespace, index: int):
    """Return ``(seed, instruction)`` for the ``index``-th evaluated episode of a task."""
    seed = args.base_seed + index * 1000
    if not args.expert_check:
        return seed, None
    found = task_env.find_solvable_seed(seed, max_tries=args.max_seed_search)
    if found is None:
        return None, None
    solved_seed, episode_info = found
    instruction = task_env.make_instruction(episode_info, args.instruction_type)
    return solved_seed, instruction


def _rollout(args, task_env: RMBenchSimEnv, policy: RMBenchPolicyClient, task: str,
             seed: int, instruction: str | None, recorder=None) -> bool:
    """Run one episode; return whether success latched."""
    obs = task_env.reset(seed, instruction=instruction)
    instruction = task_env.instruction
    policy.reset()

    frame = encode_obs(obs)
    policy.observe(frame)
    if recorder is not None:
        recorder.add(frame)
    state = obs_state(obs)

    step_lim = task_env.step_lim
    # ceil(step_lim / execute_horizon) decisions drains the budget; a small guard
    # protects against an env that never reports done.
    hard_bound = max(1, -(-step_lim // max(1, args.execute_horizon))) + 2

    decisions = 0
    while not task_env.done and decisions < hard_bound:
        decisions += 1
        chunk = policy.predict_chunk(instruction, state)
        if _LOG_SUBTASK:
            print(f"[subtask] task={task} seed={seed} dec={decisions} :: {policy.last_subtask}", flush=True)
        for action in chunk[: args.execute_horizon]:
            task_env.take_action_dense(action, substeps=args.dense_substeps)
            obs = task_env.get_obs()
            frame = encode_obs(obs)
            policy.observe(frame)
            if recorder is not None:
                recorder.add(frame)
            state = obs_state(obs)
            if task_env.done:
                break
    return task_env.success


def evaluate_task(args, task: str, policy: RMBenchPolicyClient) -> tuple[list[bool], dict]:
    """Evaluate one task over its seed schedule; return per-seed successes + metadata."""
    task_env = RMBenchSimEnv(task, args.task_config, args.rmbench_path)

    video_dir = None
    quota = None
    if args.video_dir:
        video_dir = os.path.join(args.video_dir, task)
        os.makedirs(video_dir, exist_ok=True)
        cap = max(0, int(args.video_max_per_task))
        quota = {"succ": cap, "fail": cap}

    successes: list[bool] = []
    n = int(args.seeds_per_task)
    for local_index in range(n):
        index = args.seed_pool_offset + local_index
        seed, instruction = _resolve_seed(task_env, args, index)
        if seed is None:
            print(f"[eval] {task}: no expert-solvable seed near {args.base_seed + index * 1000} -- skipped.",
                  flush=True)
            continue

        recorder = _maybe_recorder(quota)
        try:
            ok = _rollout(args, task_env, policy, task, seed, instruction, recorder=recorder)
        except Exception as exc:  # noqa: BLE001 - one bad seed must not kill the task
            import traceback
            print(f"[eval] {task} seed {seed} failed: {exc}\n{traceback.format_exc()[:800]}", flush=True)
            continue
        successes.append(ok)
        _save_video(recorder, video_dir, task, seed, ok, quota)

    rate = float(np.mean(successes)) if successes else None
    task_env.close()
    return successes, {"rate": rate, "n": len(successes), "n_success": int(sum(successes))}


def _maybe_recorder(quota):
    if not quota or (quota.get("succ", 0) <= 0 and quota.get("fail", 0) <= 0):
        return None
    from evaluation.rmbench.video_writer import RolloutVideoRecorder
    from deployment.model_server import memvla_recipe as recipe

    return RolloutVideoRecorder(fps=recipe.NATIVE_FPS)


def _save_video(recorder, video_dir, task, seed, ok, quota) -> None:
    if recorder is None or video_dir is None or len(recorder) == 0:
        return
    bucket = "succ" if ok else "fail"
    if not quota or quota.get(bucket, 0) <= 0:
        return
    out_path = os.path.join(video_dir, f"{task}__seed{seed}__{bucket}.mp4")
    try:
        if recorder.save(out_path):
            quota[bucket] -= 1
            print(f"[eval] saved {bucket} video: {out_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 - a video write must never abort the sweep
        print(f"[eval] WARNING: video save failed for {task} seed{seed}: {exc}", flush=True)


def _print_summary(results: dict, tasks: list[str], out_json: str | None) -> None:
    valid = [results[t]["rate"] for t in tasks if results.get(t, {}).get("rate") is not None]
    macro = float(np.mean(valid)) if valid else 0.0
    print("\n==== RMBench closed-loop success rates ====")
    for task in tasks:
        entry = results.get(task, {})
        rate = entry.get("rate")
        n = entry.get("n", 0)
        if rate is None:
            print(f"  {task:24s} {'N/A':>6s} (n={n})")
        else:
            print(f"  {task:24s} {rate * 100:5.1f}% (n={n})")
    excluded = sum(1 for task in tasks if results.get(task, {}).get("rate") is None)
    note = f"  [excluded {excluded} N/A task(s)]" if excluded else ""
    print(f"  {'OVERALL (macro avg)':24s} {macro * 100:5.1f}%{note}", flush=True)

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        payload = {
            "tasks": {
                task: {"success_rate": results.get(task, {}).get("rate"),
                       "n": int(results.get(task, {}).get("n", 0)),
                       "n_success": int(results.get(task, {}).get("n_success", 0))}
                for task in tasks
            },
            "macro_avg": macro,
            "n_excluded_na": excluded,
        }
        with open(out_json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[eval] wrote results JSON -> {out_json}", flush=True)


def main() -> None:
    args = parse_args()
    tasks = args.tasks if args.tasks else DEFAULT_TASKS

    # HistoryBuffer camera order must match recipe.CAMERA_LABELS: head, left, right.
    camera_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    policy = RMBenchPolicyClient(
        camera_keys=camera_keys,
        host=args.host,
        port=args.port,
        response_timeout=args.response_timeout,
    )

    results: dict[str, dict] = {}
    for task in tasks:
        _, meta = evaluate_task(args, task, policy)
        results[task] = meta
        rate = meta["rate"]
        shown = "N/A" if rate is None else f"{rate * 100:.1f}%"
        print(f"[eval] {task}: success_rate={shown} (n={meta['n']})", flush=True)

    policy.close()
    _print_summary(results, tasks, args.results_json)
    # Hard-exit so SAPIEN's renderer teardown (segfaults on exit) cannot trash the
    # exit code now that results are written; the OS reclaims the sim env.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
