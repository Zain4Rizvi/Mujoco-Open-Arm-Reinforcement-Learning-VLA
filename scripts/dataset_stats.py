"""Dataset sanity: action ranges, lengths, color-pair histogram."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from openarm_vla.data import open_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    args = p.parse_args()
    root = Path(args.dataset)
    ds = open_dataset(root)
    lengths = [int(e["length"]) for e in ds.meta.episodes]
    rows = json.loads((root / "openarm_seeds.json").read_text(encoding="utf-8"))
    pairs = Counter((r["ball_color"], r["bin_color"]) for r in rows)
    a = ds.meta.stats["action"]
    stats = {
        "n_episodes": len(lengths),
        "n_frames": sum(lengths),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": float(np.mean(lengths)),
        "action_min": np.asarray(a["min"]).tolist(),
        "action_max": np.asarray(a["max"]).tolist(),
        "action_mean": np.asarray(a["mean"]).tolist(),
        "action_std": np.asarray(a["std"]).tolist(),
        "color_pairs": {f"{b}->{k}": n for (b, k), n in sorted(pairs.items())},
        "n_unique_pairs": len(pairs),
    }
    print(json.dumps(stats, indent=2))
    if len(pairs) < 2 and len(lengths) > 5:
        print("WARNING: color pairs look collapsed; check independent randomization.")


if __name__ == "__main__":
    main()
