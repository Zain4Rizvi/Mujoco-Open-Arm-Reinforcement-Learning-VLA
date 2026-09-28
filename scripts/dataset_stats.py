"""Dataset sanity: action ranges, lengths, color-pair histogram."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    args = p.parse_args()
    root = Path(args.dataset)
    episodes = []
    for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    actions = []
    lengths = []
    pairs = Counter()
    for ep in episodes:
        data = np.load(root / "data" / f"episode_{ep['episode_index']:06d}.npz", allow_pickle=True)
        actions.append(data["action"])
        lengths.append(int(ep["length"]))
        pairs[(ep.get("ball_color"), ep.get("bin_color"))] += 1
    a = np.concatenate(actions, axis=0)
    stats = {
        "n_episodes": len(episodes),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": float(np.mean(lengths)),
        "action_min": a.min(axis=0).tolist(),
        "action_max": a.max(axis=0).tolist(),
        "action_mean": a.mean(axis=0).tolist(),
        "action_std": a.std(axis=0).tolist(),
        "color_pairs": {f"{b}->{k}": n for (b, k), n in sorted(pairs.items())},
        "n_unique_pairs": len(pairs),
    }
    print(json.dumps(stats, indent=2))
    if len(pairs) < 2 and len(episodes) > 5:
        print("WARNING: color pairs look collapsed; check independent randomization.")


if __name__ == "__main__":
    main()
