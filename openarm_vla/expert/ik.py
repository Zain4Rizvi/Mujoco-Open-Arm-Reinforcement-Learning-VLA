from __future__ import annotations

import mujoco
import numpy as np


def point_on_body(data: mujoco.MjData, body_id: int, offset: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset


def _jac6(model, data, body_id, point, dof_ids, rot_weight):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return np.vstack([jacp[:, dof_ids], rot_weight * jacr[:, dof_ids]])


def dls_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    offset: np.ndarray,
    dof_ids: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    iters: int = 150,
    damping: float = 1e-4,
    max_dq: float = 0.12,
    rot_weight: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Damped least squares IK on a point fixed to `body_id` (pos + orientation). Mutates data.

    Returns (7 joint qpos, final position error in m).
    """
    target = np.asarray(target_pos, dtype=np.float64)
    qadr = np.array([int(model.jnt_qposadr[model.dof_jntid[d]]) for d in dof_ids], dtype=int)
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        p = point_on_body(data, body_id, offset)
        r = data.xmat[body_id].reshape(3, 3)
        err_p = target - p
        err_r = 0.5 * sum(np.cross(r[:, i], target_rot[:, i]) for i in range(3))
        if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-3:
            break
        j = _jac6(model, data, body_id, p, dof_ids, rot_weight)
        e = np.concatenate([err_p, rot_weight * err_r])
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(6), e)
        data.qpos[qadr] = data.qpos[qadr] + np.clip(dq, -max_dq, max_dq)
        _clamp_qpos(model, data, dof_ids, qadr)
    mujoco.mj_forward(model, data)
    return data.qpos[qadr].copy(), float(np.linalg.norm(target - point_on_body(data, body_id, offset)))


def jacobian_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    offset: np.ndarray,
    dof_ids: np.ndarray,
    v_world: np.ndarray,
    damping: float = 1e-4,
) -> np.ndarray:
    """Joint velocity giving linear velocity v_world at the body point with zero angular velocity."""
    mujoco.mj_forward(model, data)
    j = _jac6(model, data, body_id, point_on_body(data, body_id, offset), dof_ids, 1.0)
    e = np.concatenate([np.asarray(v_world, dtype=np.float64), np.zeros(3)])
    return j.T @ np.linalg.solve(j @ j.T + damping * np.eye(6), e)


def _clamp_qpos(model, data, dof_ids, qadr) -> None:
    for d, qa in zip(dof_ids, qadr):
        j = int(model.dof_jntid[d])
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)
