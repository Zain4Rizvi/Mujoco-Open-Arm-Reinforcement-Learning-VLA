from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class EnvConfig:
    scene_xml: str = ""
    control_hz: int = 50
    n_substeps: int = 20
    image_size: int = 256
    max_episode_steps: int = 400
    success_hold_s: float = 0.5
    ball_xy_jitter: float = 0.03
    bin_xy_jitter: float = 0.04
    cam_pos_jitter: float = 0.02
    light_jitter: float = 0.15
    render: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        d = load_yaml(path)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
