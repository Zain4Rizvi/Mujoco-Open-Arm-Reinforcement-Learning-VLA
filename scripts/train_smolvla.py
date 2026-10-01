"""Fine-tune lerobot/smolvla_base on a local dataset via lerobot's own trainer (output dir must be new).

Adds on top of lerobot: <out>/train_log.csv (every step), <out>/val_log.csv (validation loss on a separate
dataset every `val_every` steps), <out>/best/pretrained_model (best validation loss), and early stopping when
validation loss stops improving. `training_state/` is pruned from all but the newest checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from openarm_vla.config import load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.data.lerobot_writer import VIDEO_BACKEND, open_dataset

# smolvla_base declares camera1/2/3; map our two cameras onto the first two.
RENAME = {
    "observation.images.image_front": "observation.images.camera1",
    "observation.images.image_wrist": "observation.images.camera2",
}
KEYS = ("dataset_root", "output_dir", "steps", "batch_size", "warmup_steps", "save_freq", "val_dataset_root",
        "val_every", "val_frames", "min_steps", "patience_evals", "min_rel_improvement")


def append_row(path: Path, header: list[str], row: list):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def float_images(batch: dict) -> dict:
    import torch

    return {k: v.float() / 255.0 if k.startswith("observation.images.") and v.dtype == torch.uint8 else v
            for k, v in batch.items()}


def prune_training_state(out: Path):
    ckpts = sorted((out / "checkpoints").glob("[0-9]*"))
    for c in ckpts[:-1]:
        shutil.rmtree(c / "training_state", ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_smolvla.yaml"))
    p.add_argument("--dataset-root")
    p.add_argument("--output-dir", help="must not exist yet")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--save-freq", type=int)
    p.add_argument("--val-dataset-root", help="separate dataset, never trained on")
    p.add_argument("--val-every", type=int)
    p.add_argument("--val-frames", type=int)
    p.add_argument("--min-steps", type=int, help="no early stop before this step")
    p.add_argument("--patience-evals", type=int, help="stop after this many validations without improvement")
    p.add_argument("--min-rel-improvement", type=float)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    for k in KEYS:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    root = Path(cfg["dataset_root"])
    out = Path(cfg["output_dir"])
    # lerobot wraps --policy.path in a Path, which turns "lerobot/smolvla_base" into "lerobot\smolvla_base" on
    # Windows and breaks the hub lookup; a local snapshot dir avoids that.
    from huggingface_hub import snapshot_download

    base = snapshot_download("lerobot/smolvla_base")
    argv = [
        "lerobot-train",
        f"--policy.path={base}",
        f"--dataset.repo_id=local/{root.name}",
        f"--dataset.root={root}",
        f"--dataset.video_backend={VIDEO_BACKEND}",
        f"--rename_map={json.dumps(RENAME)}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        f"--policy.optimizer_lr={cfg['lr']}",
        f"--policy.scheduler_warmup_steps={cfg['warmup_steps']}",
        f"--policy.scheduler_decay_steps={cfg['steps']}",
        f"--batch_size={cfg['batch_size']}",
        f"--steps={cfg['steps']}",
        f"--save_freq={cfg['save_freq']}",
        "--log_freq=10",
        f"--num_workers={cfg['num_workers']}",
        f"--output_dir={out}",
        "--wandb.enable=false",
    ]
    print(" ".join(argv))
    if args.dry_run:
        return

    if sys.platform == "win32":  # block idle sleep while this process runs; released on exit
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    import numpy as np
    import torch
    import lerobot.scripts.lerobot_train as lt
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    make = lt.make_policy
    lt.make_policy = lambda *a, **k: make(*a, **k).float()  # GTX 16xx has no bf16 compute; the VLM loads as bf16
    lt.update_last_checkpoint = lambda d: d  # Path.symlink_to needs Windows Developer Mode

    procs = {}
    make_pp = lt.make_pre_post_processors

    def make_pp_capture(*a, **k):
        procs["pre"], procs["post"] = make_pp(*a, **k)
        return procs["pre"], procs["post"]

    lt.make_pre_post_processors = make_pp_capture

    vroot = Path(cfg["val_dataset_root"])
    st = {"step": 0, "best": float("inf"), "bad": 0, "val": None}

    def val_batches(policy):
        dt = resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(f"local/{vroot.name}", root=vroot))
        ds = open_dataset(vroot, delta_timestamps=dt)
        idx = np.linspace(0, len(ds) - 1, min(int(cfg["val_frames"]), len(ds))).astype(int)
        bs = int(cfg["batch_size"])
        return [torch.utils.data.default_collate([ds[int(i)] for i in idx[j : j + bs]]) for j in range(0, len(idx), bs)]

    def validate(policy, step):
        if st["val"] is None:
            st["val"] = val_batches(policy)
        policy.eval()
        losses = []
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)  # flow matching samples noise and time; fixed seed keeps passes comparable
            for b in st["val"]:
                losses.append(policy.forward(procs["pre"](float_images(b)))[0].item())
        policy.train()
        val = float(np.mean(losses))
        improved = val < st["best"] * (1 - float(cfg["min_rel_improvement"]))
        append_row(out / "val_log.csv", ["step", "val_loss", "improved"], [step, val, int(improved)])
        if improved:
            st["best"], st["bad"] = val, 0
            best = out / "best" / "pretrained_model"
            policy.save_pretrained(best)
            procs["pre"].save_pretrained(best)
            procs["post"].save_pretrained(best)
            (out / "best" / "step.txt").write_text(str(step), encoding="utf-8")
        else:
            st["bad"] += 1
        print(f"[val] step {step}: val_loss={val:.4f} best={st['best']:.4f} improved={improved}", flush=True)
        prune_training_state(out)
        if step >= int(cfg["min_steps"]) and st["bad"] >= int(cfg["patience_evals"]):
            msg = (f"early stop at step {step}: no >{cfg['min_rel_improvement']:.0%} val-loss improvement for "
                   f"{st['bad']} validations (best {st['best']:.4f})")
            (out / "early_stop.txt").write_text(msg, encoding="utf-8")
            print(msg, flush=True)
            raise SystemExit(0)

    upd = lt.update_policy

    def update_policy(tracker, policy, *a, **k):
        tracker, out_dict = upd(tracker, policy, *a, **k)
        st["step"] += 1
        m = tracker.metrics
        out.mkdir(parents=True, exist_ok=True)
        append_row(out / "train_log.csv", ["step", "loss", "lr", "grad_norm"],
                   [st["step"], m["loss"].val, m["lr"].val, m["grad_norm"].val])
        if st["step"] % int(cfg["val_every"]) == 0:
            validate(policy, st["step"])
        return tracker, out_dict

    lt.update_policy = update_policy
    sys.argv = argv
    try:
        lt.main()
    finally:
        prune_training_state(out)


if __name__ == "__main__":
    main()
