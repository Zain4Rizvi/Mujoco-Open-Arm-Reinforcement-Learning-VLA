import json
from pathlib import Path

import numpy as np

from openarm_vla.data.lerobot_writer import LeRobotWriter


def test_lerobot_writer_layout(tmp_path):
    w = LeRobotWriter(tmp_path, fps=50)
    frame = {
        "state": np.zeros(15, np.float32),
        "action": np.zeros(8, np.float32),
        "image_front": np.zeros((16, 16, 3), np.uint8),
        "image_wrist": np.zeros((16, 16, 3), np.uint8),
    }
    w.add_episode([frame, frame], "throw the red ball into the blue bucket", {"ball_color": "red", "bin_color": "blue"})
    w.finalize()
    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["features"]["action"]["shape"] == [8]
    assert (tmp_path / "data" / "episode_000000.npz").exists()
    assert (tmp_path / "videos" / "image_front" / "episode_000000.mp4").exists()
    assert "red" in (tmp_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8")
