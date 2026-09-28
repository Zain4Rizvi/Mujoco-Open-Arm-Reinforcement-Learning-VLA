from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from openarm_vla.config import load_yaml
from openarm_vla.constants import GRASP_OFFSET, GRIPPER_CLOSED, GRIPPER_OPEN, THROW_READY_RIGHT
from openarm_vla.env.throw_env import ThrowEnv
from openarm_vla.expert.ballistic import quintic_interp, release_velocity
from openarm_vla.expert.ik import dls_ik, jacobian_velocity

AIM_Z = 0.16  # bin rim (0.13) + ball radius: the plane the ball must cross over the bin centre


@dataclass
class ExpertConfig:
    approach_height: float = 0.10
    grasp_dz: float = 0.0
    lift_height: float = 0.20
    release_forward: float = 0.10
    release_z: float = 0.70
    windup_back: float = 0.15
    windup_drop: float = 0.06
    throw_duration_s: float = 0.30
    flight_time: float = 0.40
    follow_through_s: float = 0.25
    grasp_close_steps: int = 10
    pregrasp_open: float = -0.55  # partial opening (~11 cm tips) so fingers miss neighbour balls
    ik_iters: int = 200
    ik_damping: float = 1e-4
    max_dq: float = 0.12
    rot_weight: float = 0.3
    aim_iters: int = 6
    aim_tol: float = 0.005

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExpertConfig:
        d = load_yaml(path)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ThrowExpert:
    """Open-loop FSM (ready, hover, grasp, lift, wind-up, throw, release, follow-through).

    Grasp/release goes through the env's grasp assist (gripper command), so a learned
    policy sees identical mechanics. The throw is aimed by shooting: the plan is replayed
    on the deterministic sim from a pre-throw snapshot and the aim point is shifted by the
    measured miss until it converges.
    """

    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self._actions: list[np.ndarray] = []
        self._i = 0
        self.release_step = -1
        self.plan_error: float | None = None

    def reset(self, env: ThrowEnv) -> None:
        cfg = self.cfg
        m, d = env.model, env.data
        self._hz = env.cfg.control_hz
        body, off = env._ee_body, np.array(GRASP_OFFSET)
        dof_ids = env._jnt_qvel
        ball = env.ball_pos(env.task["ball_body"])
        bin_pos = env.bin_pos(int(env.task["bin_idx"]))
        target = np.array([bin_pos[0], bin_pos[1], AIM_Z])

        saved = env.get_sim_state()
        d.qpos[env._jnt_qpos] = THROW_READY_RIGHT
        mujoco.mj_forward(m, d)
        rot_down = d.xmat[body].reshape(3, 3).copy()
        lift_p = ball + [0.0, 0.0, cfg.lift_height]
        heading = target[:2] - lift_p[:2]
        heading /= np.linalg.norm(heading) + 1e-8
        x0 = np.arctan2(rot_down[1, 0], rot_down[0, 0])  # world yaw of local x at throw_ready
        q_ready = np.asarray(THROW_READY_RIGHT, dtype=np.float64)

        def rot(yaw):
            c, s = np.cos(yaw - x0), np.sin(yaw - x0)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot_down

        def chain(points, yaws):
            """IK through `points` from throw_ready for the first yaw that reaches all of them."""
            best = (np.inf, None)
            for yaw in yaws:
                d.qpos[env._jnt_qpos] = q_ready
                qs, err = [], 0.0
                for p in points:
                    q, e = dls_ik(m, d, body, off, dof_ids, p, rot(yaw), cfg.ik_iters, cfg.ik_damping, cfg.max_dq, cfg.rot_weight)
                    qs.append(q)
                    err = max(err, e)
                if err < best[0]:
                    best = (err, qs)
                if err < 0.005:
                    break
            self.ik_err.append(round(best[0], 4))
            return best[1]

        def wrap(a):
            return (a - x0 + np.pi) % (2 * np.pi) - np.pi + x0

        # Grasp yaw: fingers (local y) as far as possible from neighbour balls.
        others = [env.ball_pos(c)[:2] for c in env._ball_body if c != env.task["ball_body"]]

        def clearance(yaw):
            fy = rot(yaw)[:2, 1]
            return min(np.linalg.norm(ball[:2] + sgn * 0.075 * fy - o) for o in others for sgn in (-1, 1))

        grasp_yaws = sorted((wrap(x0 + a) for a in np.linspace(-np.pi, np.pi, 36, endpoint=False)), key=clearance, reverse=True)
        # Throw yaw: ball leaves through the open side (local +x or -x), not through a finger.
        h = np.arctan2(heading[1], heading[0])
        throw_yaws = sorted((wrap(h), wrap(h + np.pi)), key=lambda a: abs(a - x0))

        p_rel = np.array([*(lift_p[:2] + cfg.release_forward * heading), cfg.release_z])
        p_wind = np.array([*(p_rel[:2] - cfg.windup_back * heading), cfg.release_z - cfg.windup_drop])

        self.ik_err = []
        q_hover, q_grasp = chain([ball + [0.0, 0.0, cfg.approach_height], ball + [0.0, 0.0, cfg.grasp_dz]], grasp_yaws)
        q_lift, q_wind, q_rel = chain([lift_p, p_wind, p_rel], throw_yaws)
        self._p_rel = p_rel
        self._q_wind, self._q_rel = q_wind, q_rel
        # Linear map release velocity -> joint velocity at q_rel.
        d.qpos[env._jnt_qpos] = q_rel
        self._v2qd = np.stack([jacobian_velocity(m, d, body, off, dof_ids, e) for e in np.eye(3)], axis=1)
        env.set_sim_state(saved)

        q0 = env.right_qpos()
        q_up = q0.copy()
        q_up[3] = 2.4  # raise the elbow first: home sits beside the balls, a direct path sweeps them
        pre: list[np.ndarray] = []
        pre += self._move(q0, q_up, 0.5, GRIPPER_CLOSED)
        pre += self._move(q_up, q_ready, 0.7, GRIPPER_CLOSED)
        pre += self._move(q_ready, q_hover, 0.6, cfg.pregrasp_open)
        pre += self._move(q_hover, q_grasp, 0.4, cfg.pregrasp_open)
        pre += [np.append(q_grasp, GRIPPER_CLOSED)] * cfg.grasp_close_steps
        pre += self._move(q_grasp, q_lift, 0.4, GRIPPER_CLOSED)
        pre += self._move(q_lift, q_wind, 0.45, GRIPPER_CLOSED)

        # Shooting: simulate the pre-throw part once, then iterate the throw from that snapshot.
        for a in pre:
            env.physics_step(a)
        snap = env.get_sim_state()
        aim = target.copy()
        best = (np.inf, None)
        self.aim_log = []
        jac, prev = np.eye(2), None  # Broyden estimate of d(landing)/d(aim)
        for _ in range(cfg.aim_iters):
            throw = self._throw(aim)
            env.set_sim_state(snap)
            cross = self._simulate(env, throw)
            if cross is None:
                break
            err = cross - target[:2]
            self.aim_log.append(err.round(3).tolist())
            if np.linalg.norm(err) < best[0]:
                best = (float(np.linalg.norm(err)), throw)
            if best[0] < cfg.aim_tol:
                break
            if prev is not None:
                da, de = aim[:2] - prev[0], err - prev[1]
                if da @ da > 1e-10:
                    jac += np.outer(de - jac @ da, da) / (da @ da)
            prev = (aim[:2].copy(), err)
            aim[:2] -= np.linalg.solve(jac, err)
        env.set_sim_state(saved)

        throw = best[1] if best[1] is not None else self._throw(target)
        self.plan_error = None if best[1] is None else best[0]
        self._actions = pre + throw
        self.release_step = len(pre) + self._release_offset
        self._i = 0

    def _move(self, a, b, dur, grip, qd0=None, qd1=None) -> list[np.ndarray]:
        """Quintic a->b sampled exactly at the control rate; excludes the start sample."""
        n = max(1, round(dur * self._hz))
        z = np.zeros_like(a)
        qs = quintic_interp(a, b, z if qd0 is None else qd0, z if qd1 is None else qd1, n / self._hz, n + 1)
        return [np.append(q, grip) for q in qs[1:]]

    def _throw(self, aim: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        v = release_velocity(self._p_rel, aim, cfg.flight_time)
        qd = self._v2qd @ v
        q_ft = self._q_rel + qd * cfg.follow_through_s * 0.5
        seg = self._move(self._q_wind, self._q_rel, cfg.throw_duration_s, GRIPPER_CLOSED, qd1=qd)
        ft = self._move(self._q_rel, q_ft, cfg.follow_through_s, GRIPPER_OPEN, qd0=qd)
        self._release_offset = len(seg)
        hold = [np.append(q_ft, GRIPPER_OPEN)] * (2 * self._hz)
        return seg + ft + hold

    def _simulate(self, env: ThrowEnv, throw: list[np.ndarray]) -> np.ndarray | None:
        """Ball xy where it descends through AIM_Z after release (None if not held at release)."""
        ball = env.task["ball_body"]
        prev = None
        for k, a in enumerate(throw):
            if k == self._release_offset and env.held_ball != ball:
                return None
            env.physics_step(a)
            p = env.ball_pos(ball)
            if k >= self._release_offset and prev is not None and prev[2] >= AIM_Z > p[2]:
                t = (prev[2] - AIM_Z) / (prev[2] - p[2])
                return prev[:2] + t * (p[:2] - prev[:2])
            prev = p
        return None

    def act(self, env: ThrowEnv) -> np.ndarray:
        a = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return a.astype(np.float32)
