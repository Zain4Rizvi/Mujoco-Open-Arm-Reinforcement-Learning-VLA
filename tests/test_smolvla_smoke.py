import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from openarm_vla.config import EnvConfig  # noqa: E402
from openarm_vla.env.throw_env import ThrowEnv  # noqa: E402
from openarm_vla.policies.smolvla import SmolVLAAdapter  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pretrained_smolvla_drives_env():
    env = ThrowEnv(EnvConfig(render=True, image_size=256), render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    policy = SmolVLAAdapter("lerobot/smolvla_base")

    t = time.time()
    chunk = policy.predict_chunk(obs)
    print(f"predict_chunk: {time.time() - t:.2f}s")
    assert chunk.shape == (16, 8) and chunk.dtype == np.float32
    assert np.isfinite(chunk).all()

    for a in chunk[:8]:
        obs, _, _, _, _ = env.step(a)
    assert np.isfinite(obs["state"]).all()
    np.testing.assert_allclose(
        env.data.ctrl[env._right_act], np.clip(a, env.action_space.low, env.action_space.high), atol=1e-6
    )
    env.close()
