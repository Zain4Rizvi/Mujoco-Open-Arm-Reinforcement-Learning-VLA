import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")
pytest.importorskip("datasets")

from openarm_vla.data import create_dataset, open_dataset  # noqa: E402


def _frame(task, v):
    rng = np.random.default_rng(v)
    return {
        "observation.images.image_front": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation.images.image_wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation.state": np.full(15, v, np.float32),
        "action": np.full(8, v, np.float32),
        "task": task,
    }


def test_dataset_roundtrip(tmp_path):
    root = tmp_path / "ds"
    ds = create_dataset(root, repo_id="local/ds")
    for i in range(3):
        ds.add_frame(_frame("throw the red ball into the blue bucket", i))
    ds.save_episode()
    ds.add_frame(_frame("discarded", 9))  # simulated failed rollout
    ds.clear_episode_buffer()
    for i in range(3):
        ds.add_frame(_frame("put the green ball in the purple bin", i))
    ds.save_episode()
    ds.finalize()

    ds = open_dataset(root)
    assert len(ds) == 6 and ds.meta.total_episodes == 2
    assert ds[0]["observation.images.image_front"].shape == (3, 256, 256)
    assert ds[0]["observation.state"].shape == (15,)
    assert ds[0]["action"].shape == (8,)
    assert ds[0]["task"] == "throw the red ball into the blue bucket"
    assert ds[3]["task"] == "put the green ball in the purple bin"
    assert len(ds.meta.stats["action"]["mean"]) == 8
