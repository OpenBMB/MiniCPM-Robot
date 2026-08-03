from __future__ import annotations

from typing import Any

import numpy as np


STATE_DIM = 80


def pad_state(state: Any) -> np.ndarray:
    """Flatten a simulator state and right-pad it to MiniCPM's state dimension."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size > STATE_DIM:
        raise ValueError(
            f"Simulator state has {values.size} values; expected at most {STATE_DIM}"
        )
    padded = np.zeros(STATE_DIM, dtype=np.float32)
    padded[: values.size] = values
    return padded
