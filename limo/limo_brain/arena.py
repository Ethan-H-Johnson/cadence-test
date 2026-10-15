"""The arena layouts shared by the toy sim, goal sampling and the Gazebo worlds.

A 5 m walled square; the robot spawns at the origin facing +x.
  "open"   walls only (``ros2/src/limo_sim/worlds/arena_open.sdf``), the default
  "boxes"  walls and five boxes (``ros2/src/limo_sim/worlds/arena_boxes.sdf``)
Change a layout here and in its world file together.
"""

from __future__ import annotations

import math

import numpy as np

HALF = 2.5  # inner wall faces at +-2.5 m
LAYOUTS = ("open", "boxes")
# (x, y, yaw, size_x, size_y) of each box in the "boxes" layout
BOXES = (
    (1.3, 0.9, 0.3, 0.5, 0.4),
    (-1.2, 1.4, 0.0, 0.6, 0.3),
    (-1.5, -1.0, 0.6, 0.4, 0.4),
    (0.8, -1.5, 0.0, 0.7, 0.3),
    (0.0, 1.0, 0.0, 0.3, 0.3),
)


def boxes(layout: str) -> tuple:
    return BOXES if layout == "boxes" else ()


def segments(layout: str = "boxes") -> np.ndarray:
    """All obstacle edges as an (n, 2, 2) array of segment end points."""
    h = HALF
    edges = [((-h, -h), (h, -h)), ((h, -h), (h, h)), ((h, h), (-h, h)), ((-h, h), (-h, -h))]
    for x, y, yaw, sx, sy in boxes(layout):
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(x + c * dx - s * dy, y + s * dx + c * dy)
                   for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return np.array(edges, float)


SEGMENTS = {name: segments(name) for name in LAYOUTS}


def clearance(x: float, y: float, layout: str = "boxes") -> float:
    """Distance from (x, y) to the nearest obstacle edge; 0 inside a box or outside the walls."""
    if abs(x) >= HALF or abs(y) >= HALF:
        return 0.0
    for bx, by, yaw, sx, sy in boxes(layout):
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * (x - bx) - s * (y - by), s * (x - bx) + c * (y - by)
        if abs(lx) < sx / 2 and abs(ly) < sy / 2:
            return 0.0
    p = np.array([x, y])
    a, b = SEGMENTS[layout][:, 0], SEGMENTS[layout][:, 1]
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / (d * d).sum(1), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())


def sample_goal(rng: np.random.Generator, robot_xy: tuple[float, float], layout: str = "open",
                margin: float = 0.4, min_distance: float = 1.2) -> tuple[float, float]:
    """A free point in the arena at least ``min_distance`` from the robot."""
    while True:
        x, y = rng.uniform(-HALF + margin, HALF - margin, 2)
        if clearance(x, y, layout) > margin and math.hypot(x - robot_xy[0], y - robot_xy[1]) >= min_distance:
            return float(x), float(y)
