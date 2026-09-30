"""Closed-loop eval: expert or VLA, optional matched seeds."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from openarm_vla.config import EnvConfig, load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.throw_expert import ExpertConfig, ThrowExpert
from openarm_vla.policies.dummy import DummyPolicy


def load_policy(name: str, checkpoint: str, expert: ThrowExpert):
    if name == "expert":
        return "expert", expert
    if name == "dummy":
        return "policy", DummyPolicy()
    if name == "smolvla":
        from openarm_vla.policies.smolvla import SmolVLAAdapter

        return "policy", SmolVLAAdapter(checkpoint)
    if name == "openpi":
        from openarm_vla.policies.openpi_pi0 import OpenPiAdapter

        return "policy", OpenPiAdapter(checkpoint)
    raise SystemExit(f"unknown policy {name}")


def run_policy(env, policy, chunk_horizon, replan_every):
    obs = env.get_obs()
    term = trunc = False
    inf = {}
    frames = [obs["image_front"]]
    buf = np.zeros((0, 8), np.float32)
    ptr = 0
    while not (term or trunc):
        if ptr >= replan_every or ptr >= len(buf):
            buf = policy.predict_chunk(obs)
            ptr = 0
        action = buf[min(ptr, len(buf) - 1)]
        ptr += 1
        obs, _, term, trunc, inf = env.step(action)
        frames.append(obs["image_front"])
    return inf, frames


def run_expert(env, expert):
    obs = env.get_obs()
    expert.reset(env)
    term = trunc = False
    inf = {}
    frames = [obs["image_front"]]
    while not (term or trunc):
        obs, _, term, trunc, inf = env.step(expert.act(env))
        frames.append(obs["image_front"])
    return inf, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--match-seeds", action="store_true")
    p.add_argument("--seeds-file", type=str, default=None, help="openarm_seeds.json from collect_demos; overrides --n-episodes/--seed")
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--eval-config", default=str(REPO_ROOT / "configs" / "eval.yaml"))
    args = p.parse_args()
    cfg = load_yaml(args.eval_config)
    n = args.n_episodes or int(cfg["n_episodes"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    policy_name = args.policy or cfg["policy"]
    video_dir = Path(args.video_dir or cfg["video_dir"])
    video_dir.mkdir(parents=True, exist_ok=True)
    match = args.match_seeds or bool(cfg.get("match_seeds"))

    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    expert = ThrowExpert(ExpertConfig.from_yaml(REPO_ROOT / "configs" / "expert.yaml"))
    kind, policy = load_policy(policy_name, args.checkpoint or cfg.get("checkpoint") or "", expert)
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n)]
    if args.seeds_file:
        seeds = [r["seed"] for r in json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))]
        n = len(seeds)

    counts = Counter()
    by_color = defaultdict(Counter)
    grasp = 0
    errors = []
    rows = []
    for i, s in enumerate(seeds):
        env.reset(seed=s)
        if kind == "expert":
            inf, frames = run_expert(env, expert)
        else:
            inf, frames = run_policy(env, policy, int(cfg["chunk_horizon"]), int(cfg["replan_every"]))
        mode = inf.get("failure_mode", "unknown")
        counts[mode] += 1
        by_color[inf["task"]["ball_color"]][mode] += 1
        if inf.get("grasping") or mode not in ("never_grasped",):
            if mode != "never_grasped":
                grasp += 1
        if "throw_error" in inf:
            errors.append(inf["throw_error"])
        rows.append({"seed": s, "mode": mode, "error": inf.get("throw_error"), **inf["task"]})
        print(f"ep {i} seed={s} {mode}")
        if mode != "success" and frames:
            imageio.mimsave(video_dir / f"fail_{i:03d}_{mode}.mp4", frames[::2], fps=25)

    if match and kind == "policy":
        print("--- expert on identical seeds ---")
        for i, s in enumerate(seeds):
            env.reset(seed=s)
            inf, _ = run_expert(env, expert)
            print(f"expert ep {i} seed={s} {inf.get('failure_mode')}")

    summary = {
        "n": n,
        "policy": policy_name,
        "success_rate": counts["success"] / n,
        "grasp_not_never": 1 - counts["never_grasped"] / n,
        "wrong_bucket_rate": counts["wrong_bucket"] / n,
        "taxonomy": dict(counts),
        "by_ball_color": {k: dict(v) for k, v in by_color.items()},
        "mean_throw_error": float(np.mean(errors)) if errors else None,
        "episodes": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    (video_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
