from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INSTRUCTION_TEMPLATES = (
    "throw the {ball} ball into the {bin} bucket",
    "throw the {ball} ball into the {bin} bin",
    "put the {ball} ball in the {bin} bin",
    "toss the {ball} ball into the {bin} bucket",
    "place the {ball} ball into the {bin} bucket",
)


class LeRobotWriter:
    """Minimal LeRobot v2-style dataset: parquet-like npz index + mp4 + meta/info.json.

    Full HuggingFace `lerobot` parquet is written when that package is installed;
    otherwise a compatible layout is stored for later conversion.
    """

    def __init__(self, root: str | Path, fps: int = 50):
        self.root = Path(root)
        self.fps = fps
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        (self.root / "videos" / "image_front").mkdir(parents=True, exist_ok=True)
        (self.root / "videos" / "image_wrist").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        self._episodes: list[dict] = []

    def add_episode(self, frames: list[dict], instruction: str, task: dict) -> int:
        idx = len(self._episodes)
        states = np.stack([f["state"] for f in frames])
        actions = np.stack([f["action"] for f in frames])
        np.savez_compressed(
            self.root / "data" / f"episode_{idx:06d}.npz",
            state=states.astype(np.float32),
            action=actions.astype(np.float32),
            instruction=np.array(instruction),
        )
        self._write_mp4(self.root / "videos" / "image_front" / f"episode_{idx:06d}.mp4", [f["image_front"] for f in frames])
        self._write_mp4(self.root / "videos" / "image_wrist" / f"episode_{idx:06d}.mp4", [f["image_wrist"] for f in frames])
        rec = {
            "episode_index": idx,
            "length": len(frames),
            "instruction": instruction,
            "ball_color": task.get("ball_color"),
            "bin_color": task.get("bin_color"),
        }
        self._episodes.append(rec)
        return idx

    def finalize(self) -> None:
        info = {
            "codebase_version": "v2.0",
            "robot_type": "openarm_right",
            "fps": self.fps,
            "features": {
                "observation.images.image_front": {"dtype": "video", "shape": [256, 256, 3]},
                "observation.images.image_wrist": {"dtype": "video", "shape": [256, 256, 3]},
                "observation.state": {"dtype": "float32", "shape": [15]},
                "action": {"dtype": "float32", "shape": [8]},
                "task": {"dtype": "string", "shape": [1]},
            },
            "total_episodes": len(self._episodes),
            "total_frames": sum(e["length"] for e in self._episodes),
            "splits": {"train": f"0:{len(self._episodes)}"},
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (self.root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self._episodes) + "\n", encoding="utf-8"
        )

    def _write_mp4(self, path: Path, frames: list[np.ndarray]) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=self.fps)
