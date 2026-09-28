from __future__ import annotations

from enum import Enum

import numpy as np


class FailureMode(str, Enum):
    SUCCESS = "success"
    NEVER_GRASPED = "never_grasped"
    DROPPED_DURING_GRASP = "dropped_during_grasp"
    MISSED = "missed"
    WRONG_BUCKET = "wrong_bucket"
    TIMEOUT = "timeout"


# Bin inner half-width from MJCF: wall inner edge at 0.05 m.
BIN_INNER_HALF = 0.05
BIN_Z_MIN = 0.02
BIN_Z_MAX = 0.14


def ball_in_bin(ball_pos: np.ndarray, bin_pos: np.ndarray) -> bool:
    d = ball_pos - bin_pos
    return (
        abs(float(d[0])) < BIN_INNER_HALF
        and abs(float(d[1])) < BIN_INNER_HALF
        and BIN_Z_MIN < float(ball_pos[2]) < BIN_Z_MAX
    )


def which_bin(ball_pos: np.ndarray, bin_positions: np.ndarray) -> int | None:
    for i, bp in enumerate(bin_positions):
        if ball_in_bin(ball_pos, bp):
            return i
    return None


class EpisodeTracker:
    def __init__(self, success_hold_steps: int):
        self.success_hold_steps = success_hold_steps
        self.hold = 0
        self.ever_grasped = False
        self.dropped_after_grasp = False
        self.was_grasping = False
        self.released = False
        self.landing_xy: np.ndarray | None = None

    def update(
        self,
        *,
        grasping: bool,
        gripper_open: bool,
        ball_pos: np.ndarray,
        target_bin_pos: np.ndarray,
        all_bin_pos: np.ndarray,
        target_bin_idx: int,
        airborne: bool,
    ) -> None:
        if gripper_open and self.ever_grasped:
            self.released = True

        if grasping:
            self.ever_grasped = True
            self.was_grasping = True
        elif self.was_grasping and not self.released:
            self.dropped_after_grasp = True
            self.was_grasping = False

        if self.released and not airborne and self.landing_xy is None and ball_pos[2] < 0.2:
            self.landing_xy = ball_pos[:2].copy()

        if ball_in_bin(ball_pos, target_bin_pos):
            self.hold += 1
        else:
            self.hold = 0

        self._ball_pos = ball_pos
        self._all_bin_pos = all_bin_pos
        self._target_bin_idx = target_bin_idx

    def is_success(self) -> bool:
        return self.hold >= self.success_hold_steps

    def classify(self, timed_out: bool) -> FailureMode:
        if self.is_success():
            return FailureMode.SUCCESS
        wrong = which_bin(self._ball_pos, self._all_bin_pos)
        if wrong is not None and wrong != self._target_bin_idx:
            return FailureMode.WRONG_BUCKET
        if not self.ever_grasped:
            return FailureMode.NEVER_GRASPED
        if self.dropped_after_grasp and not self.released:
            return FailureMode.DROPPED_DURING_GRASP
        if self.released:
            return FailureMode.MISSED
        if timed_out:
            return FailureMode.TIMEOUT
        return FailureMode.MISSED
