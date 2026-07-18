from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _scene_id(episode: Any) -> str:
    if isinstance(episode, dict):
        return str(episode.get("scene_id", ""))
    return str(getattr(episode, "scene_id", ""))


def split_episodes(
    episodes: Sequence[Any],
    split_count: int,
    split_id: int,
    seed: int,
) -> List[Any]:
    """Returns one deterministic split; split_count is the remainder split id."""

    if split_count <= 0:
        raise ValueError("split_count must be positive")
    if split_id < 0 or split_id > split_count:
        raise ValueError("split_id must be in [0, split_count], including remainder")
    if len(episodes) < split_count:
        raise ValueError("the dataset has fewer episodes than requested splits")

    split_size = len(episodes) // split_count
    selected_count = split_size * split_count
    random_state = np.random.RandomState(seed)
    selected = random_state.choice(
        len(episodes), selected_count, replace=False
    ).tolist()

    by_scene: Dict[str, List[int]] = {}
    for episode_index in selected:
        by_scene.setdefault(_scene_id(episodes[episode_index]), []).append(episode_index)
    ordered = [index for scene_indices in by_scene.values() for index in scene_indices]

    if split_id == split_count:
        used = set(ordered)
        return [episode for index, episode in enumerate(episodes) if index not in used]

    start = split_id * split_size
    return [episodes[index] for index in ordered[start : start + split_size]]
