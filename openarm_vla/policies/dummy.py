from __future__ import annotations

import numpy as np

from openarm_vla.constants import ACTION_DIM, GRIPPER_OPEN
from openarm_vla.expert.throw_expert import ThrowExpert


class ExpertPolicy:
    """Policy adapter around the scripted expert (for eval apples-to-apples)."""

    chunk_size = 1

    def __init__(self, expert: ThrowExpert):
        self.expert = expert

    def predict_chunk(self, obs: dict) -> np.ndarray:
        raise RuntimeError("ExpertPolicy must be stepped via act(env), not predict_chunk")

    def act(self, env) -> np.ndarray:
        return self.expert.act(env)


class DummyPolicy:
    """Zero-motion open-gripper policy for wiring tests without weights."""

    chunk_size = 16

    def predict_chunk(self, obs: dict) -> np.ndarray:
        state = obs["state"]
        q = np.concatenate([state[:7], [GRIPPER_OPEN]])
        return np.tile(q.astype(np.float32), (self.chunk_size, 1))[:, :ACTION_DIM]
