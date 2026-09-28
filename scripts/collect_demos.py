"""Collect successful expert rollouts into a LeRobot-style dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from openarm_vla.config import EnvConfig, load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.throw_expert import ExpertConfig, ThrowExpert
from openarm_vla.data.lerobot_writer import INSTRUCTION_TEMPLATES, LeRobotWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-success", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--env-config", type=str, default=str(REPO_ROOT / "configs" / "env.yaml"))
    p.add_argument("--expert-config", type=str, default=str(REPO_ROOT / "configs" / "expert.yaml"))
    p.add_argument("--dataset-config", type=str, default=str(REPO_ROOT / "configs" / "dataset.yaml"))
    args = p.parse_args()
    ds = load_yaml(args.dataset_config)
    n_success = args.n_success or int(ds["n_success"])
    max_attempts = args.max_attempts or int(ds["max_attempts"])
    out = Path(args.out or ds["out_dir"])
    seed = args.seed if args.seed is not None else int(ds["seed"])

    env = ThrowEnv(EnvConfig.from_yaml(args.env_config), render_mode="rgb_array")
    expert = ThrowExpert(ExpertConfig.from_yaml(args.expert_config))
    writer = LeRobotWriter(out, fps=env.cfg.control_hz)
    rng = __import__("numpy").random.default_rng(seed)
    got = 0
    for attempt in range(max_attempts):
        if got >= n_success:
            break
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        expert.reset(env)
        frames = []
        term = trunc = False
        inf = {}
        while not (term or trunc):
            action = expert.act(env)
            frames.append(
                {
                    "state": obs["state"],
                    "action": action,
                    "image_front": obs["image_front"],
                    "image_wrist": obs["image_wrist"],
                }
            )
            obs, _, term, trunc, inf = env.step(action)
        if inf.get("failure_mode") == "success":
            tmpl = rng.choice(INSTRUCTION_TEMPLATES)
            instruction = tmpl.format(ball=env.task["ball_color"], bin=env.task["bin_color"])
            writer.add_episode(frames, instruction, env.task)
            got += 1
            print(f"saved {got}/{n_success} (attempt {attempt + 1})")
        else:
            print(f"skip {inf.get('failure_mode')} (attempt {attempt + 1})")
    writer.finalize()
    env.close()
    print(f"wrote {got} episodes to {out}")


if __name__ == "__main__":
    main()
