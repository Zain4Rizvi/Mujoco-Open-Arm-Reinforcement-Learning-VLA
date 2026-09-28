"""Dump a few frames from a stored episode as PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=str, default="artifacts/viz_episode")
    args = p.parse_args()
    root = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = np.load(root / "data" / f"episode_{args.episode:06d}.npz", allow_pickle=True)
    vid = imageio.mimread(root / "videos" / "image_front" / f"episode_{args.episode:06d}.mp4")
    print("instruction:", npz["instruction"], "frames:", len(vid), "action shape:", npz["action"].shape)
    for i in (0, len(vid) // 2, len(vid) - 1):
        imageio.imwrite(out / f"frame_{i:04d}.png", vid[i])
    print("wrote", out)


if __name__ == "__main__":
    main()
