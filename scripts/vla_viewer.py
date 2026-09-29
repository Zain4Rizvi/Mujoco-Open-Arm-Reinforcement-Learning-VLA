"""Watch a VLA drive the right arm in the MuJoCo viewer; type instructions in this terminal.

Type a sentence + Enter to replace the instruction (the policy replans immediately),
`r` to reset to a new scene, `q` to quit. Episodes auto-reset when they end.
The env's own target (printed at reset) is what success is scored against.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

import mujoco.viewer

from openarm_vla.config import EnvConfig, load_yaml
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.policies.smolvla import SmolVLAAdapter


def read_stdin(cmds: queue.Queue):
    try:
        while True:
            cmds.put(input().strip())
    except EOFError:
        cmds.put("q")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="lerobot/smolvla_base")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="rgb_array")
    replan_every = int(load_yaml(REPO_ROOT / "configs" / "eval.yaml")["replan_every"])
    print(f"loading {args.checkpoint} ...")
    policy = SmolVLAAdapter(args.checkpoint)
    dt = 1.0 / env.cfg.control_hz

    cmds: queue.Queue = queue.Queue()
    threading.Thread(target=read_stdin, args=(cmds,), daemon=True).start()

    seed = args.seed
    prompt = None
    obs, _ = env.reset(seed=seed)
    print(f"[seed {seed}] env task: {env.task['instruction']}")
    print("type an instruction + Enter, 'r' = reset, 'q' = quit")
    buf, ptr = None, 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            cmd = None if cmds.empty() else cmds.get()
            if cmd == "q":
                break
            if cmd == "r":
                seed += 1
                obs, _ = env.reset(seed=seed)
                buf = None
                print(f"[seed {seed}] env task: {env.task['instruction']}")
            elif cmd:
                prompt = cmd
                buf = None
                print(f"instruction -> {prompt!r}")

            if buf is None or ptr >= replan_every:
                if prompt:
                    obs["instruction"] = prompt
                buf, ptr = policy.predict_chunk(obs), 0  # physics paused while the VLA thinks
            obs, _, term, trunc, info = env.step(buf[ptr])
            ptr += 1
            viewer.sync()
            time.sleep(dt)

            if term or trunc:
                print(f"episode end: {info['failure_mode']}")
                seed += 1
                obs, _ = env.reset(seed=seed)
                buf = None
                print(f"[seed {seed}] env task: {env.task['instruction']}")
    env.close()


if __name__ == "__main__":
    main()
