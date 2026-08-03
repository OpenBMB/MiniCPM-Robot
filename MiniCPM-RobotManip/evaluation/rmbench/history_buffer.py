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

"""Per-camera strided history buffer for memVLA closed-loop evaluation.

Each decision step needs the recent history video per camera at the exact frame
counts and strides the checkpoint was trained on. The buffer stores every observed
native-cadence frame and, on demand, slices out the strided window per view.

Episode-start clamping repeats the first frame for offsets that reach before the
episode began, matching training's ``clamp_vision_offsets_to_episode``.
"""

from __future__ import annotations

import numpy as np

from deployment.model_server import memvla_recipe as recipe


class HistoryBuffer:
    """Store observed frames per camera and emit the trained strided windows.

    Camera order follows ``recipe.CAMERA_LABELS``; the caller observes a dict keyed
    by the short simulator camera names it was constructed with.
    """

    def __init__(self, camera_keys):
        camera_keys = list(camera_keys)
        if len(camera_keys) != recipe.NUM_VIEWS:
            raise ValueError(
                f"expected {recipe.NUM_VIEWS} cameras (order matches "
                f"{list(recipe.CAMERA_LABELS)}), got {camera_keys}"
            )
        self.camera_keys = camera_keys
        self.view_specs = recipe.VIEW_SPECS
        self._buffers: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}

    def reset(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()

    def observe(self, frames: dict[str, np.ndarray]) -> None:
        missing = [key for key in self.camera_keys if key not in frames]
        if missing:
            raise KeyError(f"observe() missing frames for cameras: {missing}")
        for key in self.camera_keys:
            self._buffers[key].append(np.asarray(frames[key], dtype=np.uint8))

    def _history(self, key: str, n_frames: int, stride: int) -> list[np.ndarray]:
        buffer = self._buffers[key]
        current = len(buffer) - 1
        if current < 0:
            raise RuntimeError("call observe() before building the history window")
        if n_frames > 1:
            offsets = [-(n_frames - 1 - i) * stride for i in range(n_frames)]
        else:
            offsets = [0]
        # Clamp to the episode start: offsets before frame 0 repeat the first frame.
        return [buffer[max(0, current + offset)] for offset in offsets]

    def build_views(self) -> list[list[np.ndarray]]:
        """Return the strided window per camera, in ``recipe.CAMERA_LABELS`` order."""
        return [
            self._history(key, n_frames, stride)
            for key, (n_frames, stride) in zip(self.camera_keys, self.view_specs)
        ]
