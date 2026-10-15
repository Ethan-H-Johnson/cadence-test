"""What the brain sees, what it can do, and what it is paid: shared by sim and real robot.

Observation (10 inputs, all in [0, 1]):
  0-7  nearness in eight lidar sectors across the LIMO's 240-degree scan, left to right
       (1 at contact, 0 at ``NEAR_RANGE`` metres or beyond)
  8    measured forward speed / ``FAST``
  9    measured turn rate magnitude / ``SPIN``

With a goal (``--task goal``), 4 more inputs:
  10   goal ahead: (1 + cos bearing) / 2
  11   goal to the left: max(0, sin bearing)
  12   goal to the right: max(0, -sin bearing)
  13   goal closeness: 1 at the goal, 0 at ``GOAL_FAR`` metres or beyond

Actions: five discrete (linear m/s, angular rad/s) commands.

Reward for the previous action, paid with the next observation (``reward``, roaming):
  forward progress this tick, scaled so cruising forward earns ~0.5,
  minus a turning cost (0.3 at full spin), so circling in the open does not pay,
  minus 1 on a bump (anything closer than ``BUMP_RANGE``),
  minus a small penalty for crowding obstacles.
``goal_reward`` instead pays the distance closed toward the goal (cruising straight at it
earns ~0.5), a smaller turning cost (0.1), the same bump penalty, and 1 on arrival.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECTORS = 8
NEAR_RANGE = 1.5
BUMP_RANGE = 0.16  # metres from the lidar; LIMO body extends ~0.1 m ahead of it
FAST = 0.3
SPIN = 1.2

ACTIONS = (
    ("forward", FAST, 0.0),
    ("veer left", 0.15, 0.8),
    ("veer right", 0.15, -0.8),
    ("spin left", 0.0, SPIN),
    ("spin right", 0.0, -SPIN),
)
INPUTS = SECTORS + 2
GOAL_INPUTS = 4
GOAL_FAR = 3.0
GOAL_REACHED = 0.3  # metres from the robot's centre


@dataclass
class Reading:
    """One sensor snapshot: scan ranges ordered from angle_min (right) to angle_max (left)."""

    ranges: np.ndarray
    v: float  # measured forward speed, m/s
    w: float  # measured yaw rate, rad/s
    x: float = 0.0
    y: float = 0.0
    t: float = 0.0  # sim or wall time, s
    tilt: float = 0.0  # degrees from upright; 180 = upside down
    yaw: float = 0.0  # heading in the arena frame, rad


FALLEN = 45.0  # degrees of tilt counted as a fall


def fallen(r: Reading) -> bool:
    return r.tilt > FALLEN


def observe(r: Reading) -> np.ndarray:
    ranges = np.nan_to_num(np.asarray(r.ranges, float), nan=NEAR_RANGE, posinf=NEAR_RANGE)
    ranges = np.clip(ranges, 0.0, NEAR_RANGE)
    # left-to-right order reads naturally; the scan runs right to left
    sectors = np.array_split(ranges[::-1], SECTORS)
    nearness = 1.0 - np.array([s.min() for s in sectors]) / NEAR_RANGE
    v, w = np.nan_to_num(r.v), np.nan_to_num(r.w)
    motion = [np.clip(v / FAST, -1, 1) * 0.5 + 0.5, min(abs(w) / SPIN, 1.0)]
    return np.concatenate([nearness, motion])[None, :]


def goal_bearing(r: Reading, goal: tuple[float, float]) -> tuple[float, float]:
    """(bearing from the robot's heading in rad, positive = left; distance in m)."""
    dx, dy = goal[0] - r.x, goal[1] - r.y
    b = np.arctan2(dy, dx) - r.yaw
    return float(np.arctan2(np.sin(b), np.cos(b))), float(np.hypot(dx, dy))


def observe_goal(r: Reading, goal: tuple[float, float]) -> np.ndarray:
    b, d = goal_bearing(r, goal)
    extra = [(1 + np.cos(b)) / 2, max(0.0, np.sin(b)), max(0.0, -np.sin(b)),
             1 - min(d / GOAL_FAR, 1.0)]
    return np.concatenate([observe(r)[0], extra])[None, :]


def bumped(r: Reading) -> bool:
    ranges = np.asarray(r.ranges, float)
    ranges = ranges[np.isfinite(ranges) & (ranges > 0.02)]  # drop self-hits and dropouts
    return bool(ranges.size and ranges.min() < BUMP_RANGE)


def reward(before: Reading, after: Reading) -> float:
    # forward distance this tick relative to cruising for the same time
    dt = max(after.t - before.t, 1e-3)
    progress = np.hypot(after.x - before.x, after.y - before.y) / (FAST * dt)
    pay = 0.5 * float(np.clip(np.sign(after.v) * progress, -1, 1))
    pay -= 0.3 * min(abs(after.w) / SPIN, 1.0)
    if bumped(after) or fallen(after):
        pay -= 1.0
    else:
        crowd = observe(after)[0, :SECTORS].max()
        pay -= 0.2 * max(crowd - 0.8, 0.0) / 0.2
    return float(np.clip(pay, -1.0, 1.0))


def goal_reward(before: Reading, after: Reading, goal: tuple[float, float]) -> tuple[float, bool]:
    """(pay, reached) for the tick from ``before`` to ``after`` toward ``goal``."""
    dt = max(after.t - before.t, 1e-3)
    closed = goal_bearing(before, goal)[1] - goal_bearing(after, goal)[1]
    pay = 0.5 * float(np.clip(closed / (FAST * dt), -1, 1))
    pay -= 0.1 * min(abs(after.w) / SPIN, 1.0)
    if bumped(after) or fallen(after):
        pay -= 1.0
    reached = goal_bearing(after, goal)[1] < GOAL_REACHED
    if reached:
        pay += 1.0
    return float(np.clip(pay, -1.0, 1.0)), reached


def command(action: int, weak_left: float = 1.0) -> tuple[float, float]:
    """The (v, w) sent to the base. ``weak_left`` < 1 emulates a weakened left side:
    the commanded wheel speeds are converted, the left side scaled, and converted back."""
    _, v, w = ACTIONS[action]
    if weak_left == 1.0:
        return v, w
    half = 0.172 / 2  # LIMO wheel separation
    left, right = v - w * half, v + w * half
    left *= weak_left
    return (left + right) / 2, (right - left) / (2 * half)


def reflex(obs: np.ndarray) -> int:
    """Hand-written control: go forward when the front is clear, else turn to the open side."""
    near = obs[0, :SECTORS]
    front = near[3:5].max()
    if front < 0.5:
        return 0
    left, right = near[:3].mean(), near[5:].mean()
    if front > 0.75:
        return 3 if left < right else 4
    return 1 if left < right else 2


def reflex_goal(obs: np.ndarray) -> int:
    """Hand-written goal seeking: pick the action that would close the distance to the goal
    fastest over the next 0.5 s, avoiding sectors with something within ~0.75 m."""
    near = obs[0, :SECTORS]  # left to right, 30 deg each across +120..-120
    ahead, left, right = obs[0, SECTORS + 2:SECTORS + 5]
    bearing = np.arctan2(left - right, 2 * ahead - 1)
    best, best_score = 0, -np.inf
    for i, (_, v, w) in enumerate(ACTIONS):
        turn = w * 0.5
        sector = int(np.clip((np.radians(120) - turn) / np.radians(30), 0, SECTORS - 1))
        # closing speed after the turn, plus a little credit for facing the goal so that
        # spins toward a goal behind the robot beat spins away from it
        score = v * np.cos(bearing - turn) + 0.1 * np.cos(bearing - turn)
        score -= 1.0 * max(near[sector] - 0.5, 0) * (v > 0)
        if score > best_score:
            best, best_score = i, score
    return best
