from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    chunk_size: int

    def predict_chunk(self, obs: dict) -> np.ndarray:
        """Return actions shaped (T, 8). Physics is paused by the caller during this call."""
        ...
