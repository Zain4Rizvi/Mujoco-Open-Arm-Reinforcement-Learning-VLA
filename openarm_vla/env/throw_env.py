from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from openarm_vla.config import EnvConfig
from openarm_vla.constants import (
    BIN_BODIES,
    COLOR_RGBA,
    COLORS,
    GRASP_OFFSET,
    GRIP_CLOSE_CMD,
    GRIPPER_OPEN,
    HOME_CTRL_LEFT,
    TRAIN_TEMPLATES,
    N_SUBSTEPS,
    REPO_ROOT,
    RIGHT_ACTUATORS,
    RIGHT_JOINTS,
    SCENE_XML,
    STATE_DIM,
)
from openarm_vla.env.success import EpisodeTracker

BIN_MIN_SEP = 0.10  # smallest nominal bin spacing (Chebyshev) in the MJCF


class ThrowEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = "rgb_array"):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        xml = Path(self.cfg.scene_xml) if self.cfg.scene_xml else SCENE_XML
        if not xml.is_absolute():
            xml = REPO_ROOT / xml
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._n_substeps = self.cfg.n_substeps or N_SUBSTEPS
        self._image_size = self.cfg.image_size
        self._renderer: mujoco.Renderer | None = None
        if self.cfg.render and render_mode != "human":
            self._renderer = mujoco.Renderer(self.model, height=self._image_size, width=self._image_size)

        self._right_act = np.array([self.model.actuator(n).id for n in RIGHT_ACTUATORS], dtype=int)
        self._left_act = np.array(
            [self.model.actuator(n).id for n in
             ("left_joint1_ctrl", "left_joint2_ctrl", "left_joint3_ctrl", "left_joint4_ctrl",
              "left_joint5_ctrl", "left_joint6_ctrl", "left_joint7_ctrl", "left_finger1_ctrl")],
            dtype=int,
        )
        self._jnt_qpos = np.array([self.model.joint(n).qposadr[0] for n in RIGHT_JOINTS], dtype=int)
        self._jnt_qvel = np.array([self.model.joint(n).dofadr[0] for n in RIGHT_JOINTS], dtype=int)
        self._finger_qpos = int(self.model.joint("openarm_right_finger_joint1").qposadr[0])
        self._site_ee = int(self.model.site("right_ee_control_point").id)
        self._ee_body = int(self.model.body("openarm_right_ee_base_link").id)
        self._grasp_offset = np.array(GRASP_OFFSET)
        self._finger_bodies = {
            int(self.model.body(n).id) for n in ("openarm_right_ee_inner_finger", "openarm_right_ee_outer_finger")
        }
        self._cam_front = int(self.model.camera("frontcam").id)
        self._cam_wrist = int(self.model.camera("camera_wrist_right").id)
        self._ball_body = {c: int(self.model.body(f"ball_{c}").id) for c in COLORS}
        self._ball_geom = {c: int(self.model.geom(f"ball_{c}_geom").id) for c in COLORS}
        self._bin_body = [int(self.model.body(n).id) for n in BIN_BODIES]
        self._bin_geoms = []
        for i in range(5):
            names = [f"bin{i}_floor", f"bin{i}_n", f"bin{i}_s", f"bin{i}_e", f"bin{i}_w"]
            self._bin_geoms.append([int(self.model.geom(n).id) for n in names])
        self._ball_qadr = []
        for c in COLORS:
            bid = self._ball_body[c]
            jnt = int(self.model.body_jntadr[bid])
            self._ball_qadr.append(int(self.model.jnt_qposadr[jnt]))
        self._eq_weld = {c: int(self.model.eq(f"grasp_right_{c}").id) for c in COLORS}
        self._home_key = int(self.model.key("home").id)
        self._bin_pos0 = self.model.body_pos[[self._bin_body[i] for i in range(5)]].copy()
        self._cam_pos0 = self.model.cam_pos[self._cam_front].copy()
        self._light_pos0 = self.model.light_pos[0].copy()
        self._light_diff0 = self.model.light_diffuse[0].copy()

        act_low = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._right_act], dtype=np.float32)
        act_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._right_act], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        h = w = self._image_size
        self.observation_space = spaces.Dict(
            {
                "image_front": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "image_wrist": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
                "instruction": spaces.Text(max_length=200),
            }
        )

        self._tracker: EpisodeTracker | None = None
        self._step_count = 0
        self.task: dict = {}
        self._rng = np.random.default_rng()

    def close(self):
        self._renderer = None
        return super().close()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        for eq in self._eq_weld.values():
            self.data.eq_active[eq] = 0

        ball_colors = list(COLORS)
        bin_colors = list(COLORS)
        self._rng.shuffle(ball_colors)
        self._rng.shuffle(bin_colors)
        target_ball = str(options.get("ball_color") or self._rng.choice(COLORS))
        target_bin_color = str(options.get("bin_color") or self._rng.choice(COLORS))
        # Map shuffled colors onto the five ball bodies (named orange..purple in XML).
        body_color = {body: col for body, col in zip(COLORS, ball_colors)}
        bin_assigned = list(bin_colors)

        for body_name in COLORS:
            self.model.geom_rgba[self._ball_geom[body_name]] = COLOR_RGBA[body_color[body_name]]
        # Jitter balls on the table; resample until no ball touches anything but the table
        # (overlapping balls explode apart, and the home wrist sits right next to the ball set).
        home_xy = self.data.qpos[[a + k for a in self._ball_qadr for k in (0, 1)]].reshape(5, 2).copy()
        ball_geoms = set(self._ball_geom.values())
        table = int(self.model.geom("table_top").id)
        for _ in range(100):
            for i, adr in enumerate(self._ball_qadr):
                j = self.cfg.ball_xy_jitter
                self.data.qpos[adr:adr + 2] = home_xy[i] + self._rng.uniform(-j, j, 2)
                self.data.qpos[adr + 2] = 0.43
                self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            if not any(
                ({c.geom1, c.geom2} & ball_geoms) and table not in (c.geom1, c.geom2)
                for c in self.data.contact[: self.data.ncon]
            ):
                break

        for i in range(5):
            col = bin_assigned[i]
            rgba = np.array(COLOR_RGBA[col], dtype=np.float64)
            floor_rgba = rgba.copy()
            floor_rgba[3] = 0.85
            ids = self._bin_geoms[i]
            self.model.geom_rgba[ids[0]] = floor_rgba
            for gid in ids[1:]:
                self.model.geom_rgba[gid] = rgba
        # Bins are 0.14 m across the walls but only ~0.10 m apart in the MJCF, so jitter can push a
        # neighbour's wall across a bin's opening. Resample until no pair is closer than nominal.
        for _ in range(1000):
            xy = self._bin_pos0[:, :2] + self._rng.uniform(-self.cfg.bin_xy_jitter, self.cfg.bin_xy_jitter, (5, 2))
            sep = np.abs(xy[:, None] - xy[None]).max(-1) + np.eye(5)
            if sep.min() >= BIN_MIN_SEP:
                break
        for i in range(5):
            self.model.body_pos[self._bin_body[i], :2] = xy[i]

        # Resolve which XML body holds the target color, and which bin index is that color.
        target_ball_body = next(b for b, c in body_color.items() if c == target_ball)
        target_bin_idx = bin_assigned.index(target_bin_color)

        self.model.cam_pos[self._cam_front] = self._cam_pos0 + self._rng.uniform(
            -self.cfg.cam_pos_jitter, self.cfg.cam_pos_jitter, 3
        )
        self.model.light_pos[0] = self._light_pos0 + self._rng.uniform(-self.cfg.light_jitter, self.cfg.light_jitter, 3)
        self.model.light_diffuse[0] = np.clip(
            self._light_diff0 * (1.0 + self._rng.uniform(-0.15, 0.15)), 0.2, 1.0
        )

        tmpl = str(options.get("instruction") or self._rng.choice(TRAIN_TEMPLATES))
        instruction = tmpl.format(ball=target_ball, bin=target_bin_color)

        self.data.ctrl[self._left_act] = HOME_CTRL_LEFT
        self.data.ctrl[self._right_act[:7]] = self.data.qpos[self._jnt_qpos]
        self.data.ctrl[self._right_act[7]] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

        hold_steps = int(self.cfg.success_hold_s * self.cfg.control_hz)
        self._tracker = EpisodeTracker(hold_steps)
        self._step_count = 0
        self.task = {
            "ball_color": target_ball,
            "bin_color": target_bin_color,
            "ball_body": target_ball_body,
            "bin_idx": target_bin_idx,
            "body_color": body_color,
            "bin_colors": bin_assigned,
            "instruction": instruction,
        }
        obs = self._obs(instruction)
        return obs, {"task": {k: v for k, v in self.task.items() if k != "body_color"}}

    def physics_step(self, action: np.ndarray) -> None:
        """One control tick: clip, substeps. No bookkeeping, so planners can reuse it."""
        action = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high)
        self.data.ctrl[self._left_act] = HOME_CTRL_LEFT
        self.data.ctrl[self._right_act] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray):
        self.physics_step(action)
        self._step_count += 1
        ball_pos = self.ball_pos(self.task["ball_body"])
        bin_pos = self.bin_pos(self.task["bin_idx"])
        all_bins = np.stack([self.bin_pos(i) for i in range(5)])
        grasping = self._is_grasping()
        gripper_open = float(self.data.ctrl[self._right_act[7]]) < GRIP_CLOSE_CMD
        airborne = ball_pos[2] > 0.2 and np.linalg.norm(self.ball_vel(self.task["ball_body"])[:2]) > 0.3
        self._tracker.update(
            grasping=grasping,
            gripper_open=gripper_open,
            ball_pos=ball_pos,
            target_bin_pos=bin_pos,
            all_bin_pos=all_bins,
            target_bin_idx=int(self.task["bin_idx"]),
            airborne=airborne,
        )
        terminated = self._tracker.is_success()
        truncated = self._step_count >= self.cfg.max_episode_steps
        mode = self._tracker.classify(truncated) if (terminated or truncated) else None
        reward = 1.0 if terminated else 0.0
        obs = self._obs(self.task["instruction"])
        info = {
            "failure_mode": None if mode is None else mode.value,
            "grasping": grasping,
            "ball_pos": ball_pos.copy(),
            "task": {"ball_color": self.task["ball_color"], "bin_color": self.task["bin_color"]},
        }
        if terminated or truncated:
            info["failure_mode"] = mode.value
            if self._tracker.landing_xy is not None:
                info["throw_error"] = float(np.linalg.norm(self._tracker.landing_xy - bin_pos[:2]))
            else:
                info["throw_error"] = float(np.linalg.norm(ball_pos[:2] - bin_pos[:2]))
        return obs, reward, terminated, truncated, info

    def get_obs(self) -> dict:
        return self._obs(self.task["instruction"])

    def _obs(self, instruction: str) -> dict:
        state = np.concatenate(
            [
                self.data.qpos[self._jnt_qpos],
                self.data.qvel[self._jnt_qvel],
                [self.data.qpos[self._finger_qpos]],
            ]
        ).astype(np.float32)
        if self._renderer is None:
            z = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
            front, wrist = z, z.copy()
        else:
            front = self._rgb(self._cam_front)
            wrist = self._rgb(self._cam_wrist)
        return {
            "image_front": front,
            "image_wrist": wrist,
            "state": state,
            "instruction": instruction,
        }

    def _rgb(self, cam_id: int) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render().copy()

    def ball_pos(self, body_name: str) -> np.ndarray:
        return self.data.xpos[self._ball_body[body_name]].copy()

    def ball_vel(self, body_name: str) -> np.ndarray:
        bid = self._ball_body[body_name]
        jnt = int(self.model.body_jntadr[bid])
        dof = int(self.model.jnt_dofadr[jnt])
        return self.data.qvel[dof:dof + 3].copy()

    def bin_pos(self, idx: int) -> np.ndarray:
        return self.data.xpos[self._bin_body[idx]].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site_ee].copy()

    def grasp_point(self) -> np.ndarray:
        return self.data.xpos[self._ee_body] + self.data.xmat[self._ee_body].reshape(3, 3) @ self._grasp_offset

    def _is_grasping(self) -> bool:
        return self.holding(self.task["ball_body"])

    def holding(self, ball_body: str) -> bool:
        """Both fingers touch the ball."""
        g = self._ball_geom[ball_body]
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            if g in (c.geom1, c.geom2):
                touching.add(int(self.model.geom_bodyid[c.geom2 if c.geom1 == g else c.geom1]))
        return self._finger_bodies <= touching

    def get_sim_state(self) -> np.ndarray:
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        s = np.zeros(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, s, spec)
        return s

    def set_sim_state(self, s: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)

    def right_qpos(self) -> np.ndarray:
        return self.data.qpos[self._jnt_qpos].copy()

    def ctrl_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        return self.action_space.low.copy(), self.action_space.high.copy()

    def plant_ball_in_bin(self, ball_body: str, bin_idx: int) -> None:
        pos = self.bin_pos(bin_idx)
        adr = self._ball_qadr[COLORS.index(ball_body)]
        self.data.qpos[adr : adr + 3] = [pos[0], pos[1], 0.05]
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
