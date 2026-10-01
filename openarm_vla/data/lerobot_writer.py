from __future__ import annotations

from pathlib import Path

from openarm_vla.constants import CONTROL_HZ, RIGHT_ACTUATORS, RIGHT_JOINTS

# torchcodec can't load on this Windows box (no shared FFmpeg); pyav decodes everywhere instead.
VIDEO_BACKEND = "pyav"

_IMG = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "observation.images.image_front": _IMG,
    "observation.images.image_wrist": _IMG,
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": [f"{j}_pos" for j in RIGHT_JOINTS] + [f"{j}_vel" for j in RIGHT_JOINTS] + ["finger_pos"],
    },
    "action": {"dtype": "float32", "shape": (8,), "names": list(RIGHT_ACTUATORS)},
}


def create_dataset(root: str | Path, repo_id: str = "local/openarm_throw"):
    """New lerobot dataset in write mode. `root` must not exist yet."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=CONTROL_HZ,
        features=FEATURES,
        root=root,
        robot_type="openarm_right",
        use_videos=True,
        image_writer_threads=4,
        video_backend=VIDEO_BACKEND,
    )


def open_dataset(root: str | Path, **kwargs):
    """Read an existing dataset (repo_id is `local/<dir name>`); kwargs go to LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    return LeRobotDataset(f"local/{root.name}", root=root, video_backend=VIDEO_BACKEND, **kwargs)
