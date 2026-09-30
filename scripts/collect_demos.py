"""Collect successful expert rollouts into a lerobot dataset (--out must not exist yet).

Parallel runs: give each process its own --out and a distinct --seed-offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openarm_vla.config import EnvConfig, load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.data import create_dataset
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.throw_expert import ExpertConfig, ThrowExpert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-success", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0, help="added to --seed; use distinct values per parallel process")
    p.add_argument("--env-config", type=str, default=str(REPO_ROOT / "configs" / "env.yaml"))
    p.add_argument("--expert-config", type=str, default=str(REPO_ROOT / "configs" / "expert.yaml"))
    p.add_argument("--dataset-config", type=str, default=str(REPO_ROOT / "configs" / "dataset.yaml"))
    args = p.parse_args()
    ds_cfg = load_yaml(args.dataset_config)
    n_success = args.n_success or int(ds_cfg["n_success"])
    max_attempts = args.max_attempts or int(ds_cfg["max_attempts"])
    out = Path(args.out or ds_cfg["out_dir"])
    seed = (args.seed if args.seed is not None else int(ds_cfg["seed"])) + args.seed_offset

    env = ThrowEnv(EnvConfig.from_yaml(args.env_config), render_mode="rgb_array")
    expert = ThrowExpert(ExpertConfig.from_yaml(args.expert_config))
    ds = create_dataset(out, repo_id=f"local/{out.name}")
    rng = np.random.default_rng(seed)
    rows = []
    for attempt in range(max_attempts):
        if len(rows) >= n_success:
            break
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        expert.reset(env)
        term = trunc = False
        inf = {}
        while not (term or trunc):
            action = expert.act(env)
            ds.add_frame(  # obs recorded BEFORE the action it precedes
                {
                    "observation.images.image_front": obs["image_front"],
                    "observation.images.image_wrist": obs["image_wrist"],
                    "observation.state": obs["state"].astype(np.float32),
                    "action": np.asarray(action).astype(np.float32),
                    "task": env.task["instruction"],
                }
            )
            obs, _, term, trunc, inf = env.step(action)
        if inf.get("failure_mode") == "success":
            ds.save_episode()
            rows.append(
                {
                    "episode_index": len(rows),
                    "seed": ep_seed,
                    "instruction": env.task["instruction"],
                    "ball_color": env.task["ball_color"],
                    "bin_color": env.task["bin_color"],
                }
            )
            print(f"saved {len(rows)}/{n_success} (attempt {attempt + 1})", flush=True)
        else:
            ds.clear_episode_buffer()
            print(f"skip {inf.get('failure_mode')} (attempt {attempt + 1})", flush=True)
    ds.finalize()
    (out / "openarm_seeds.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    env.close()
    print(f"wrote {len(rows)} episodes to {out}")


if __name__ == "__main__":
    main()
