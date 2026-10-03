"""Eval-only chunk smoother. k=3 on a spike uses the edge sample, not zeros."""

import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_policy.py"
_spec = importlib.util.spec_from_file_location("eval_policy", _path)
eval_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_policy)


def test_smooth_chunk_edge_not_zero():
    actions = np.array([[0.0], [10.0], [0.0]], dtype=np.float32)
    out = eval_policy.smooth_chunk(actions, 3)
    assert out.shape == (3, 1)
    assert np.allclose(out, 10.0 / 3.0)


def test_smooth_chunk_off_is_identity():
    actions = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert eval_policy.smooth_chunk(actions, 0) is actions
    assert eval_policy.smooth_chunk(actions, 1) is actions
