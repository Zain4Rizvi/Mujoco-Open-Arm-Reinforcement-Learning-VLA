"""Fine-tune lerobot/smolvla_base on a local dataset via lerobot's own trainer (output dir must be new)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openarm_vla.config import load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.data.lerobot_writer import VIDEO_BACKEND

# smolvla_base declares camera1/2/3; map our two cameras onto the first two.
RENAME = {
    "observation.images.image_front": "observation.images.camera1",
    "observation.images.image_wrist": "observation.images.camera2",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_smolvla.yaml"))
    p.add_argument("--dataset-root")
    p.add_argument("--output-dir", help="must not exist yet")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    for k in ("dataset_root", "output_dir", "steps", "batch_size", "warmup_steps"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    root = Path(cfg["dataset_root"])
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
        f"--save_freq={cfg['steps']}",
        "--log_freq=10",
        f"--num_workers={cfg['num_workers']}",
        f"--output_dir={cfg['output_dir']}",
        "--wandb.enable=false",
    ]
    print(" ".join(argv))
    if args.dry_run:
        return

    import lerobot.scripts.lerobot_train as lt

    make = lt.make_policy
    lt.make_policy = lambda *a, **k: make(*a, **k).float()  # GTX 16xx has no bf16 compute; the VLM loads as bf16
    lt.update_last_checkpoint = lambda d: d  # Path.symlink_to needs Windows Developer Mode
    sys.argv = argv
    lt.main()


if __name__ == "__main__":
    main()
