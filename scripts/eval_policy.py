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


def rollout(env, act):
    """Run act(obs) -> action to episode end; also track ever-grasped and closest gripper-to-ball distance."""
    obs = env.get_obs()
    term = trunc = False
    inf = {}
    frames = [obs["image_front"]]
    grasped, min_d = False, float("inf")
    while not (term or trunc):
        obs, _, term, trunc, inf = env.step(act(obs))
        frames.append(obs["image_front"])
        grasped |= bool(inf.get("grasping"))
        min_d = min(min_d, float(np.linalg.norm(env.grasp_point() - env.ball_pos(env.task["ball_body"]))))
    return inf, frames, grasped, min_d


def run_policy(env, policy, chunk_horizon, replan_every):
    buf = np.zeros((0, 8), np.float32)
    ptr = 0

    def act(obs):
        nonlocal buf, ptr
        if ptr >= replan_every or ptr >= len(buf):
            buf = policy.predict_chunk(obs)
            ptr = 0
        ptr += 1
        return buf[min(ptr - 1, len(buf) - 1)]

    return rollout(env, act)


def run_expert(env, expert):
    expert.reset(env)
    return rollout(env, lambda obs: expert.act(env))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--match-seeds", action="store_true")
    p.add_argument("--seeds-file", type=str, default=None, help="openarm_seeds.json from collect_demos; overrides --n-episodes/--seed")
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--save-all-videos", action="store_true", help="default: failures only")
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
    # Stored instructions are replayed verbatim: the template list may have changed since collection.
    reset_opts = [{} for _ in range(n)]
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n)]
    if args.seeds_file:
        file_rows = json.loads(Path(args.seeds_file).read_text(encoding="utf-8"))
        seeds = [r["seed"] for r in file_rows]
        reset_opts = [{"instruction": r["instruction"]} for r in file_rows]
        n = len(seeds)

    counts = Counter()
    by_color = defaultdict(Counter)
    errors = []
    rows = []
    for i, s in enumerate(seeds):
        env.reset(seed=s, options=reset_opts[i])
        if kind == "expert":
            inf, frames, grasped, min_d = run_expert(env, expert)
        else:
            inf, frames, grasped, min_d = run_policy(env, policy, int(cfg["chunk_horizon"]), int(cfg["replan_every"]))
        mode = inf.get("failure_mode", "unknown")
        counts[mode] += 1
        by_color[inf["task"]["ball_color"]][mode] += 1
        if "throw_error" in inf:
            errors.append(inf["throw_error"])
        rows.append({"seed": s, "mode": mode, "error": inf.get("throw_error"), "grasped": grasped,
                     "min_grasp_dist": min_d, "instruction": env.task["instruction"], **inf["task"]})
        print(f"ep {i} seed={s} {mode} grasped={grasped} min_grasp_dist={100 * min_d:.1f}cm", flush=True)
        if (args.save_all_videos or mode != "success") and frames:
            task = f"{inf['task']['ball_color']}-into-{inf['task']['bin_color']}"
            imageio.mimsave(video_dir / f"{i:03d}_{mode}_{task}.mp4", frames[::2], fps=25)

    if match and kind == "policy":
        print("--- expert on identical seeds ---")
        for i, s in enumerate(seeds):
            env.reset(seed=s, options=reset_opts[i])
            inf, *_ = run_expert(env, expert)
            print(f"expert ep {i} seed={s} {inf.get('failure_mode')}")

    summary = {
        "n": n,
        "policy": policy_name,
        "success_rate": counts["success"] / n,
        "grasp_not_never": 1 - counts["never_grasped"] / n,
        "grasp_rate": sum(r["grasped"] for r in rows) / n,
        "mean_min_grasp_dist_cm": 100 * float(np.mean([r["min_grasp_dist"] for r in rows])),
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
