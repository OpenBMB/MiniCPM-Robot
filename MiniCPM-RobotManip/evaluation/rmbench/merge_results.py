# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Merge per-GPU RMBench result tables into one macro-average summary.

Each driver writes ``results_gpu<N>.json``. Entries for the same task are pooled by
summing successes and seed counts, so the merge is correct whether GPUs were split
by task (each task in one file) or by seed (each task in every file, each a disjoint
seed slice). The macro average follows the reported benchmark metric: ``macro9``
excludes ``place_block_mat`` (to match the RMBench figure other models report);
``macro10`` over all tasks is also recorded. Tasks with zero pooled seeds are
excluded from both.

Standard-library only.
"""

from __future__ import annotations

import argparse
import json

# macro9 excludes place_block_mat, matching the reported RMBench metric.
MACRO9_EXCLUDE = "place_block_mat"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("results", nargs="*", help="Per-GPU results_gpu<N>.json files")
    args = parser.parse_args()

    # Pool successes + seed counts per task across all shards.
    pooled: dict[str, dict[str, int]] = {}
    for path in args.results:
        try:
            with open(path) as handle:
                data = json.load(handle)
        except FileNotFoundError:
            print(f"[merge] WARNING: missing result file {path}")
            continue
        for task, entry in data.get("tasks", {}).items():
            n = int(entry.get("n", 0))
            rate = entry.get("success_rate")
            # Prefer the exact success count; fall back to rate*n for older tables.
            n_success = entry.get("n_success")
            if n_success is None:
                n_success = round(rate * n) if (rate is not None and n) else 0
            bucket = pooled.setdefault(task, {"n": 0, "n_success": 0})
            bucket["n"] += n
            bucket["n_success"] += int(n_success)

    tasks: dict[str, dict] = {
        task: {
            "success_rate": (b["n_success"] / b["n"]) if b["n"] else None,
            "n": b["n"],
            "n_success": b["n_success"],
        }
        for task, b in pooled.items()
    }

    def macro(exclude: set) -> tuple[float, int]:
        rates = [
            entry["success_rate"]
            for task, entry in tasks.items()
            if task not in exclude and entry.get("success_rate") is not None
        ]
        return (sum(rates) / len(rates) if rates else 0.0), len(rates)

    macro9, n9 = macro({MACRO9_EXCLUDE})
    macro10, n10 = macro(set())

    payload = {
        "tasks": tasks,
        "macro9": macro9,
        "macro9_n_tasks": n9,
        "macro10": macro10,
        "macro10_n_tasks": n10,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("\n==== RMBench closed-loop summary ====")
    for task in sorted(tasks):
        entry = tasks[task]
        rate = entry.get("success_rate")
        n = entry.get("n", 0)
        shown = "N/A" if rate is None else f"{rate * 100:5.1f}%"
        flag = "  (excluded from macro9)" if task == MACRO9_EXCLUDE else ""
        print(f"  {task:24s} {shown:>6s} (n={n}){flag}")
    print(f"  {'MACRO9 (avg)':24s} {macro9 * 100:5.1f}% over {n9} tasks")
    print(f"  {'MACRO10 (avg)':24s} {macro10 * 100:5.1f}% over {n10} tasks", flush=True)


if __name__ == "__main__":
    main()
