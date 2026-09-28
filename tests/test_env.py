from pathlib import Path

import numpy as np
import pytest

from openarm_vla.config import EnvConfig
from openarm_vla.env.success import EpisodeTracker, FailureMode, ball_in_bin
from openarm_vla.env.throw_env import ThrowEnv


def test_ball_in_bin_geometry():
    bin_pos = np.array([0.5, 0.0, 0.0])
    assert ball_in_bin(np.array([0.5, 0.0, 0.05]), bin_pos)
    assert not ball_in_bin(np.array([0.8, 0.0, 0.05]), bin_pos)


def test_tracker_success_and_wrong_bucket():
    t = EpisodeTracker(success_hold_steps=3)
    bins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for _ in range(3):
        t.update(
            grasping=False,
            gripper_open=True,
            ball_pos=np.array([0.0, 0.0, 0.05]),
            target_bin_pos=bins[0],
            all_bin_pos=bins,
            target_bin_idx=0,
            airborne=False,
        )
    assert t.is_success()
    assert t.classify(False) is FailureMode.SUCCESS

    t2 = EpisodeTracker(2)
    t2.update(
        grasping=True,
        gripper_open=False,
        ball_pos=np.array([1.0, 0.0, 0.05]),
        target_bin_pos=bins[0],
        all_bin_pos=bins,
        target_bin_idx=0,
        airborne=False,
    )
    t2.update(
        grasping=False,
        gripper_open=True,
        ball_pos=np.array([1.0, 0.0, 0.05]),
        target_bin_pos=bins[0],
        all_bin_pos=bins,
        target_bin_idx=0,
        airborne=False,
    )
    assert t2.classify(True) is FailureMode.WRONG_BUCKET


def test_reset_seed_reproducible():
    cfg = EnvConfig(render=False, max_episode_steps=5)
    env = ThrowEnv(cfg, render_mode=None)
    o1, i1 = env.reset(seed=0)
    o2, i2 = env.reset(seed=0)
    assert i1["task"]["ball_color"] == i2["task"]["ball_color"]
    assert i1["task"]["bin_color"] == i2["task"]["bin_color"]
    np.testing.assert_allclose(o1["state"], o2["state"], atol=1e-6)
    env.close()


def test_action_clip_and_state_dim():
    cfg = EnvConfig(render=False)
    env = ThrowEnv(cfg, render_mode=None)
    obs, _ = env.reset(seed=1)
    assert obs["state"].shape == (15,)
    assert obs["instruction"]
    huge = env.action_space.high + 10
    obs2, _, term, trunc, info = env.step(huge)
    assert obs2["state"].shape == (15,)
    np.testing.assert_allclose(env.data.ctrl[env._right_act], env.action_space.high, atol=1e-6)
    env.close()


def test_planted_success():
    cfg = EnvConfig(render=False, success_hold_s=0.1, max_episode_steps=20)
    env = ThrowEnv(cfg, render_mode=None)
    obs, info = env.reset(seed=2)
    env.plant_ball_in_bin(env.task["ball_body"], env.task["bin_idx"])
    action = np.clip(env.data.ctrl[env._right_act], env.action_space.low, env.action_space.high)
    done = False
    mode = None
    for _ in range(20):
        _, r, term, trunc, inf = env.step(action)
        if term or trunc:
            mode = inf["failure_mode"]
            done = term
            break
    assert mode == "success"
    assert done
    env.close()


def test_phase0_rgb_artifact(tmp_path):
    cfg = EnvConfig(render=True, image_size=256)
    env = ThrowEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=3)
    assert obs["image_front"].shape == (256, 256, 3)
    assert obs["image_wrist"].shape == (256, 256, 3)
    assert obs["image_front"].std() > 5 and obs["image_wrist"].std() > 5
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    from imageio.v2 import imwrite

    imwrite(out / "phase0_rgb.png", np.concatenate([obs["image_front"], obs["image_wrist"]], axis=1))
    assert (out / "phase0_rgb.png").exists()
    env.close()
