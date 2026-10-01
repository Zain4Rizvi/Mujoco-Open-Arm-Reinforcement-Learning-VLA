"""Closed-loop eval (all videos) of every checkpoint of a run plus best/, on the same seeds; then re-plot.

Writes <out>/step_NNNNNN/ and <out>/best/ (videos + eval_summary.json). Already-evaluated dirs are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training output dir")
    p.add_argument("--seeds-file", required=True)
    p.add_argument("--out", required=True, help="e.g. artifacts/stageA")
    p.add_argument("--baseline", help="eval_summary.json passed through to plot_training.py")
    args = p.parse_args()
    if sys.platform == "win32":  # block idle sleep while this process runs; released on exit
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    run, out = Path(args.run), Path(args.out)
    ckpts = [(f"step_{c.name}", c / "pretrained_model") for c in sorted((run / "checkpoints").glob("[0-9]*"))]
    ckpts.append(("best", run / "best" / "pretrained_model"))
    for name, ckpt in ckpts:
        if not (ckpt / "config.json").exists() or (out / name / "eval_summary.json").exists():
            continue
        print(f"== eval {name}", flush=True)
        subprocess.run([sys.executable, "scripts/eval_policy.py", "--policy", "smolvla", "--checkpoint", str(ckpt),
                        "--seeds-file", args.seeds_file, "--save-all-videos", "--video-dir", str(out / name)])
    plot = [sys.executable, "scripts/plot_training.py", "--run", str(run), "--evals", str(out)]
    subprocess.run(plot + (["--baseline", args.baseline] if args.baseline else []))


if __name__ == "__main__":
    main()
