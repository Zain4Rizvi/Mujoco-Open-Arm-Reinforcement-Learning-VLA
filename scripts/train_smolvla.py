"""SmolVLA LoRA fine-tune launcher. Dry-runs if lerobot/torch are missing."""

from __future__ import annotations

import argparse
from pathlib import Path

from openarm_vla.config import load_yaml
from openarm_vla.constants import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_smolvla.yaml"))
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if args.overfit:
        cfg["overfit"] = True
        cfg["steps"] = min(int(cfg.get("steps", 500)), 500)
        cfg["n_episodes"] = 10
    print("SmolVLA train config:", cfg)
    print("Expected input: cameras image_front/image_wrist, state dim 15, action dim 8, chunk", cfg.get("chunk_size"))
    if args.dry_run:
        print("dry-run ok (no GPU training)")
        return
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    except ImportError:
        print("torch not installed — cannot train. Re-run with GPU env after: uv pip install -e '.[train]'")
        return
    try:
        import lerobot
    except ImportError:
        print("lerobot not installed — uv pip install -e '.[train]'")
        return
    print(
        "Launch training with lerobot's smolvla recipe, pointing --dataset.repo_id at",
        cfg["dataset_repo"],
        "output",
        cfg["output_dir"],
        "lora_rank",
        cfg.get("lora_rank"),
    )
    print("See README.md for the exact lerobot CLI (API varies by lerobot version).")


if __name__ == "__main__":
    main()
