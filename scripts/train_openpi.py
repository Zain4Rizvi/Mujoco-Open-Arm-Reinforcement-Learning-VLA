"""Optional openpi/pi0 LoRA launch notes as a runnable dry-run."""

from __future__ import annotations

import argparse

from openarm_vla.constants import ACTION_DIM, STATE_DIM


def main():
    argparse.ArgumentParser(description="Print openpi mapping for this robot").parse_args()
    print("openpi custom data config for OpenArm throw:")
    print(f"  state_dim={STATE_DIM} action_dim={ACTION_DIM} chunk_horizon=16")
    print("  cameras: image_front, image_wrist")
    print("  replan_every=8  (pause physics during inference)")
    print("Install: git clone https://github.com/Physical-Intelligence/openpi && follow their uv/conda docs")
    print("Use their LoRA example config; replace robot data transforms with the dims above.")
    try:
        import openpi  # noqa: F401

        print("openpi import: ok")
    except ImportError:
        print("openpi import: missing (expected unless you installed it)")


if __name__ == "__main__":
    main()
