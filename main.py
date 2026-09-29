"""Play a few scripted throws in the MuJoCo viewer."""

import time

import mujoco.viewer

from openarm_vla.config import EnvConfig
from openarm_vla.constants import REPO_ROOT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.throw_expert import ExpertConfig, ThrowExpert

N_SAMPLES = 10


def main():
    env = ThrowEnv(EnvConfig.from_yaml(REPO_ROOT / "configs" / "env.yaml"), render_mode="human")
    expert = ThrowExpert(ExpertConfig.from_yaml(REPO_ROOT / "configs" / "expert.yaml"))
    dt = 1.0 / env.cfg.control_hz
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(N_SAMPLES):
            if not viewer.is_running():
                break
            _, info = env.reset(seed=i)
            viewer.sync()
            expert.reset(env)
            term = trunc = False
            while viewer.is_running() and not (term or trunc):
                _, _, term, trunc, info = env.step(expert.act(env))
                viewer.sync()
                time.sleep(dt)
            print(f"sample {i}: {info.get('failure_mode')}  {env.task['instruction']}")
            hold_until = time.time() + 1.5
            while viewer.is_running() and time.time() < hold_until:
                viewer.sync()
                time.sleep(dt)
    env.close()


if __name__ == "__main__":
    main()
