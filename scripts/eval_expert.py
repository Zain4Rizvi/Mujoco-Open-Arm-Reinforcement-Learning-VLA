"""Evaluate the scripted expert; write videos and a taxonomy table."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from openarm_vla.config import EnvConfig
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.throw_expert import ExpertConfig, ThrowExpert


def rollout(env, expert, record=False):
    obs = env.get_obs()
    expert.reset(env)
    frames = []
    term = trunc = False
    inf = {}
    while not (term or trunc):
        if record:
            frames.append(obs["image_front"])
        action = expert.act(env)
        obs, _, term, trunc, inf = env.step(action)
        if record:
            frames.append(obs["image_front"])
    return inf, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-dir", type=str, default="artifacts/expert")
    p.add_argument("--max-videos", type=int, default=6)
    p.add_argument("--env-config", default=str(REPO_ROOT / "configs" / "env.yaml"))
    p.add_argument("--expert-config", default=str(REPO_ROOT / "configs" / "expert.yaml"))
    args = p.parse_args()
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = ThrowEnv(EnvConfig.from_yaml(args.env_config), render_mode="rgb_array")
    expert = ThrowExpert(ExpertConfig.from_yaml(args.expert_config))
    rng = np.random.default_rng(args.seed)
    counts = Counter()
    errors = []
    n_vid = {"success": 0, "fail": 0}
    for i in range(args.n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        want_vid = n_vid["success"] < args.max_videos // 2 or n_vid["fail"] < args.max_videos // 2
        inf, frames = rollout(env, expert, record=want_vid)
        mode = inf.get("failure_mode", "unknown")
        counts[mode] += 1
        if "throw_error" in inf:
            errors.append(inf["throw_error"])
        print(f"ep {i}: {mode} err={inf.get('throw_error')}")
        if frames and ((mode == "success" and n_vid["success"] < args.max_videos // 2) or (mode != "success" and n_vid["fail"] < args.max_videos // 2)):
            tag = "ok" if mode == "success" else "fail"
            imageio.mimsave(video_dir / f"{tag}_{i:03d}_{mode}.mp4", frames[::2], fps=25)
            n_vid["success" if mode == "success" else "fail"] += 1
    n = args.n_episodes
    summary = {
        "n": n,
        "success_rate": counts["success"] / n,
        "taxonomy": dict(counts),
        "mean_throw_error": float(np.mean(errors)) if errors else None,
    }
    print(json.dumps(summary, indent=2))
    (video_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
