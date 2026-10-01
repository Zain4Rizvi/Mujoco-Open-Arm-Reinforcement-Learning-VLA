"""Dump frames from a stored episode as PNGs; with --replay, re-run its actions in the env and check they match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from openarm_vla.config import EnvConfig
from openarm_vla.constants import REPO_ROOT
from openarm_vla.data import open_dataset
from openarm_vla.env.throw_env import ThrowEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=str, default="artifacts/viz_episode")
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    root = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = open_dataset(root)
    ep = ds.meta.episodes[args.episode]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    task = ep["tasks"][0]
    print("instruction:", task, "frames:", hi - lo)
    for i in (lo, (lo + hi) // 2, hi - 1):
        item = ds[i]
        for cam in ("image_front", "image_wrist"):
            img = (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            imageio.imwrite(out / f"ep{args.episode}_{cam}_{i - lo:04d}.png", img)
    print("wrote", out)
    if not args.replay:
        return

    rows = json.loads((root / "openarm_seeds.json").read_text(encoding="utf-8"))
    row = rows[args.episode]
    assert row["episode_index"] == args.episode
    sub = ds.hf_dataset.select(range(lo, hi))
    states = np.asarray(sub["observation.state"], np.float32)
    actions = np.asarray(sub["action"], np.float32)
    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    obs, _ = env.reset(seed=row["seed"], options={"instruction": row["instruction"]})
    assert env.task["instruction"] == task, (env.task["instruction"], task)
    video0 = ds[lo]["observation.images.image_front"].permute(1, 2, 0).numpy()
    print("frame-0 mean |env render - decoded video|:", float(np.abs(obs["image_front"] / 255.0 - video0).mean()))
    worst = 0.0
    inf = {}
    for t in range(hi - lo):
        worst = max(worst, float(np.abs(obs["state"] - states[t]).max()))
        obs, _, term, trunc, inf = env.step(actions[t])
        if term or trunc:
            break
    print("max |env state - dataset state|:", worst)
    assert worst < 1e-4, "replay diverged from recorded states"
    print("replay result:", inf.get("failure_mode"))
    env.close()


if __name__ == "__main__":
    main()
