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

"""Adapt memVLA history inference to the starVLA policy contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from deployment.model_server import memvla_recipe as recipe


_IGNORED_STARVLA_KWARGS = frozenset(
    {
        "do_sample",
        "use_ddim",
        "num_ddim_steps",
        "cfg_scale",
    }
)


def _to_numpy(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "cpu"):
        action = action.cpu()
    if hasattr(action, "numpy"):
        action = action.numpy()
    return np.asarray(action)


class MemVLAPolicyWrapper:
    """Expose a history-conditioned memVLA runner as a starVLA policy.

    Unlike the single-frame wrapper, one request carries a window of history per
    camera rather than a single time step. Frame counts are fixed by the training
    recipe, so a request with the wrong shape is rejected instead of silently
    truncated.
    """

    def __init__(self, runner: Any, default_embodiment_id: int = recipe.EMBODIMENT_ID) -> None:
        self._runner = runner
        self.action_chunk_size = recipe.ACTION_HORIZON
        self.action_dim = recipe.UNIFIED_DIM
        self.state_dim = recipe.UNIFIED_DIM
        self.max_num_embodiments = int(
            getattr(getattr(runner, "model", None), "action_head", None).max_num_embodiments
        )
        self.default_embodiment_id = self._validate_embodiment_id(default_embodiment_id)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model-invariant fields sent during the WebSocket handshake."""
        return {
            "action_chunk_size": self.action_chunk_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "max_num_embodiments": self.max_num_embodiments,
            "default_embodiment_id": self.default_embodiment_id,
            "training_obs_image_size": list(recipe.IMAGE_SIZE),
            "max_batch_size": 1,
            "single_frame": False,
            "multi_view": True,
            "history": True,
            "subtask": True,
            "camera_labels": list(recipe.CAMERA_LABELS),
            "view_specs": [list(spec) for spec in recipe.VIEW_SPECS],
            "control_frequency_hz": recipe.CONTROL_FREQUENCY_HZ,
            "action_normalization": "none",
            "actions_ready_for_execution": True,
            "available_unnorm_keys": [],
            "default_unnorm_key": None,
        }

    def _validate_embodiment_id(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("embodiment_id must be an integer")
        embodiment_id = int(value)
        if not 0 <= embodiment_id < self.max_num_embodiments:
            raise ValueError(
                f"embodiment_id must be in [0, {self.max_num_embodiments - 1}], "
                f"got {embodiment_id}"
            )
        return embodiment_id

    @staticmethod
    def _validate_seed(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("seed must be an integer or None")
        return int(value)

    @staticmethod
    def _extract_text(example: dict[str, Any]) -> str:
        lang = example.get("lang")
        text = example.get("text")
        if lang is not None and text is not None and lang != text:
            raise ValueError("example.lang and example.text disagree")
        prompt = lang if lang is not None else text
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("example.lang (or example.text) must be a non-empty string")
        return prompt

    @staticmethod
    def _extract_views(example: dict[str, Any]) -> list[list[np.ndarray]]:
        views = example.get("views")
        if not isinstance(views, (list, tuple)) or not views:
            raise TypeError("example.views must be a non-empty sequence of per-camera frame lists")
        if len(views) != recipe.NUM_VIEWS:
            raise ValueError(
                f"example.views must contain {recipe.NUM_VIEWS} cameras "
                f"(order: {list(recipe.CAMERA_LABELS)}); got {len(views)}"
            )

        validated: list[list[np.ndarray]] = []
        for view_index, view in enumerate(views):
            if isinstance(view, np.ndarray) and view.ndim == 4:
                view = list(view)
            elif not isinstance(view, (list, tuple)):
                raise TypeError(f"example.views[{view_index}] must be a frame sequence")
            frames = []
            for frame_index, frame in enumerate(view):
                if not isinstance(frame, np.ndarray):
                    raise TypeError(
                        f"example.views[{view_index}][{frame_index}] must be a NumPy array"
                    )
                if frame.dtype != np.uint8:
                    raise ValueError(
                        f"example.views[{view_index}][{frame_index}] must have dtype uint8, "
                        f"got {frame.dtype}"
                    )
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError(
                        f"example.views[{view_index}][{frame_index}] must have shape HxWx3, "
                        f"got {frame.shape}"
                    )
                frames.append(frame)
            validated.append(frames)

        recipe.validate_views([len(view) for view in validated])
        return validated

    def _extract_state(self, example: dict[str, Any]) -> np.ndarray | None:
        state = example.get("state")
        if state is None:
            return None
        state_array = np.asarray(state, dtype=np.float32)
        valid_shapes = {
            (self.state_dim,),
            (1, self.state_dim),
            (1, 1, self.state_dim),
        }
        if state_array.shape not in valid_shapes:
            raise ValueError(
                f"state must have shape ({self.state_dim},), "
                f"(1, {self.state_dim}), or (1, 1, {self.state_dim}); "
                f"got {state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state must contain only finite values")
        return state_array

    def predict_action(
        self,
        examples: list[dict[str, Any]],
        unnorm_key: str | None = None,
        embodiment_id: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one history window and return execution-ready actions plus the sub-task."""
        del unnorm_key  # Compatibility-only; MiniCPM actions need no unnormalization.

        unknown = set(kwargs) - _IGNORED_STARVLA_KWARGS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported inference arguments: {names}")
        ignored = sorted(name for name in kwargs if kwargs[name] is not None)
        if ignored:
            logging.debug("Ignoring starVLA-only inference arguments: %s", ignored)

        if not isinstance(examples, list) or len(examples) != 1:
            raise ValueError("examples must be a list containing exactly one window")
        example = examples[0]
        if not isinstance(example, dict):
            raise TypeError("examples[0] must be a dict")

        views = self._extract_views(example)
        text = self._extract_text(example)
        state = self._extract_state(example)
        selected_embodiment = self._validate_embodiment_id(
            self.default_embodiment_id if embodiment_id is None else embodiment_id
        )
        selected_seed = self._validate_seed(seed)

        action, subtask = self._runner.predict(
            views=views,
            instruction=text,
            state=state,
            embodiment_id=selected_embodiment,
            seed=selected_seed,
        )
        action_array = _to_numpy(action)
        expected_shape = (self.action_chunk_size, self.action_dim)
        if action_array.shape != expected_shape:
            raise ValueError(
                f"memVLA returned action shape {action_array.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(action_array.dtype, np.floating):
            raise TypeError(f"memVLA returned non-floating actions: {action_array.dtype}")
        if not np.isfinite(action_array).all():
            raise ValueError("memVLA returned non-finite actions")

        # The runner already returns float32 execution-ready actions. astype(copy=False)
        # only enforces the wire dtype; values, dimensions, and ordering are untouched.
        action_array = action_array.astype(np.float32, copy=False)
        return {"actions": action_array[None, ...], "subtask": str(subtask)}
