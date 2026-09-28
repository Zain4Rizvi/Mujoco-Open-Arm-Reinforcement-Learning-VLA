from __future__ import annotations

import numpy as np


def release_velocity(p_release: np.ndarray, p_target: np.ndarray, flight_time: float, g: float = 9.81) -> np.ndarray:
    """World-frame linear velocity that lands a projectile at p_target after flight_time."""
    d = np.asarray(p_target, dtype=np.float64) - np.asarray(p_release, dtype=np.float64)
    t = float(flight_time)
    return np.array([d[0] / t, d[1] / t, d[2] / t + 0.5 * g * t])


def quintic_interp(q0: np.ndarray, q1: np.ndarray, qd0: np.ndarray, qd1: np.ndarray, T: float, n: int) -> np.ndarray:
    """n samples of a quintic from (q0,qd0) to (q1,qd1) over duration T. First sample is q0."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    qd0 = np.asarray(qd0, dtype=np.float64)
    qd1 = np.asarray(qd1, dtype=np.float64)
    ts = np.linspace(0.0, T, n)
    out = np.zeros((n, q0.shape[0]))
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    for i, t in enumerate(ts):
        # q = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
        a0 = q0
        a1 = qd0
        a2 = np.zeros_like(q0)
        a3 = (20 * (q1 - q0) - (8 * qd1 + 12 * qd0) * T) / (2 * T3)
        a4 = (30 * (q0 - q1) + (14 * qd1 + 16 * qd0) * T) / (2 * T4)
        a5 = (12 * (q1 - q0) - (6 * qd1 + 6 * qd0) * T) / (2 * T5)
        out[i] = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    return out
