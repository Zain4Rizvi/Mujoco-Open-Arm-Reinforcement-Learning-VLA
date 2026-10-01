"""Plot a training run to <run>/train_loss.png: loss, lr, grad norm, and per-checkpoint closed-loop evals.

--evals points at the eval_checkpoints.py output (step_NNNNNN/ and best/ subdirs with eval_summary.json);
--baseline adds a reference eval_summary.json as dashed lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    """Trailing mean over the last w steps (fewer at the start)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - w, 0)
    return (c[i] - c[lo]) / (i - lo)


def load_evals(evals: Path, run: Path) -> list[tuple[int, str, dict]]:
    res = []
    for d in sorted(evals.glob("*/eval_summary.json")):
        name = d.parent.name
        if m := re.fullmatch(r"step_(\d+)", name):
            res.append((int(m.group(1)), name, json.loads(d.read_text(encoding="utf-8"))))
        elif name == "best" and (run / "best" / "step.txt").exists():
            res.append((int((run / "best" / "step.txt").read_text()), name, json.loads(d.read_text(encoding="utf-8"))))
    return sorted(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training output dir (has train_log.csv)")
    p.add_argument("--evals", help="dir with step_*/eval_summary.json from eval_checkpoints.py")
    p.add_argument("--baseline", help="eval_summary.json to draw as a reference")
    args = p.parse_args()
    run = Path(args.run)
    tr, va = read_csv(run / "train_log.csv"), read_csv(run / "val_log.csv")
    evals = load_evals(Path(args.evals), run) if args.evals else []
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    a.plot(tr["step"], tr["loss"], alpha=0.25, lw=0.6, label="train (raw)")
    a.plot(tr["step"], smooth(tr["loss"]), lw=1.5, label="train (mean of last 50)")
    if va:
        a.plot(va["step"], va["val_loss"], "o-", color="C3", label="validation")
        best = va["improved"] > 0
        a.plot(va["step"][best], va["val_loss"][best], "*", ms=12, color="C3", label="saved best/")
    if (run / "early_stop.txt").exists():
        a.axvline(tr["step"][-1], color="k", ls=":", label="early stop")
    a.set(yscale="log", xlabel="step", ylabel="flow-matching loss", title="Loss")
    a.legend()
    ax[0, 1].plot(tr["step"], tr["lr"])
    ax[0, 1].set(xlabel="step", title="Learning rate")
    ax[1, 0].plot(tr["step"], tr["grad_norm"], alpha=0.3, lw=0.6)
    ax[1, 0].plot(tr["step"], smooth(tr["grad_norm"]), lw=1.5)
    ax[1, 0].set(yscale="log", xlabel="step", title="Gradient norm (raw, mean of last 50)")

    a = ax[1, 1]
    if evals:
        steps = [s for s, n, _ in evals if n != "best"]
        per = [e for _, n, e in evals if n != "best"]
        a.plot(steps, [100 * e["grasp_rate"] for e in per], "o-", label="grasp rate %")
        a.plot(steps, [100 * e["success_rate"] for e in per], "s-", label="success %")
        b = a.twinx()
        b.plot(steps, [e["mean_min_grasp_dist_cm"] for e in per], "^-", color="C2", label="closest gripper-ball (cm)")
        b.set_ylabel("cm")
        for s, n, e in evals:
            if n == "best":
                a.plot([s], [100 * e["grasp_rate"]], "*", ms=14, color="C0", label="best/ grasp rate %")
                b.plot([s], [e["mean_min_grasp_dist_cm"]], "*", ms=14, color="C2", label="best/ closest (cm)")
        if base:
            a.axhline(100 * base["grasp_rate"], color="C0", ls="--", lw=0.8, label="baseline grasp %")
            b.axhline(base["mean_min_grasp_dist_cm"], color="C2", ls="--", lw=0.8, label="baseline closest (cm)")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = b.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=8)
        a.set(xlabel="checkpoint step", ylabel="%", ylim=(-5, 105), title=f"Closed-loop eval (n={per[0]['n'] if per else '?'})")
    else:
        a.text(0.5, 0.5, "no checkpoint evals yet", ha="center", va="center", transform=a.transAxes)
        a.set_axis_off()
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(run / "train_loss.png", dpi=110)
    print("wrote", run / "train_loss.png")
    for s, n, e in evals:
        print(f"{n:>12} step {s:>6}: success {e['success_rate']:.0%} grasp {e['grasp_rate']:.0%} "
              f"closest {e['mean_min_grasp_dist_cm']:.1f} cm")


if __name__ == "__main__":
    main()
