"""Worlds the controller can drive.

``ToySim``  a pure-NumPy 2D LIMO in a walled arena, lockstep: time advances only when the
            controller says so, so emulated compute latency costs simulated time exactly.
``Bridge``  the ROS bridge node (Gazebo or the real robot) over TCP, paced by its clock.

Both implement ``reading()``, ``tick(old, new, latency_s, period_s)``, ``reset()``, ``close()``.
``tick`` models a serial control loop: the previous command keeps driving the base while the
brain computes for ``latency_s``; the new command then drives it until the tick ends, at
``max(period_s, latency_s)``.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

from . import arena
from .robot import Reading

WHEEL_SEPARATION = 0.172
BODY_RADIUS = 0.12
SCAN_MIN, SCAN_MAX, SCAN_RAYS, SCAN_RANGE = -2.0944, 2.0944, 240, 8.0


class ToySim:
    """2D LIMO with a first-order velocity lag like a real base.

    ``layout="open"`` (default) or ``"boxes"`` are the Gazebo arenas (``arena.py``), spawning
    at random free poses; ``layout="random"`` is a ``size`` m square with ``obstacles`` random
    boxes."""

    def __init__(self, seed: int = 0, size: float = 4.0, obstacles: int = 5, lag_s: float = 0.15,
                 layout: str = "open"):
        self.rng = np.random.default_rng(seed)
        self.lag = lag_s
        if layout in arena.LAYOUTS:
            self.size = 2 * arena.HALF
            seg = arena.SEGMENTS[layout]
        else:
            self.size = size
            h = size / 2
            walls = [((-h, -h), (h, -h)), ((h, -h), (h, h)), ((h, h), (-h, h)), ((-h, h), (-h, -h))]
            for _ in range(obstacles):
                cx, cy = self.rng.uniform(-h + 0.6, h - 0.6, 2)
                a, b = self.rng.uniform(0.15, 0.35, 2)
                c = [(cx - a, cy - b), (cx + a, cy - b), (cx + a, cy + b), (cx - a, cy + b)]
                walls += [(c[i], c[(i + 1) % 4]) for i in range(4)]
            seg = np.array(walls, float)
        self.p0, self.p1 = seg[:, 0], seg[:, 1]
        self.angles = np.linspace(SCAN_MIN, SCAN_MAX, SCAN_RAYS)
        self.bumps = 0
        self.reset()

    # geometry -----------------------------------------------------------
    def _clearance(self, x: float, y: float) -> float:
        p = np.array([x, y])
        d = self.p1 - self.p0
        t = np.clip(((p - self.p0) * d).sum(1) / (d * d).sum(1), 0, 1)
        return float(np.linalg.norm(self.p0 + t[:, None] * d - p, axis=1).min())

    def _scan(self) -> np.ndarray:
        ang = self.yaw + self.angles
        r = np.stack([np.cos(ang), np.sin(ang)], 1)  # rays
        d = self.p1 - self.p0  # segments
        q = self.p0 - np.array([self.x, self.y])
        cross = r[:, None, 0] * d[None, :, 1] - r[:, None, 1] * d[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q[None, :, 0] * d[None, :, 1] - q[None, :, 1] * d[None, :, 0]) / cross
            u = (q[None, :, 0] * r[:, None, 1] - q[None, :, 1] * r[:, None, 0]) / cross
        hit = (np.abs(cross) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf).min(1)
        t = np.minimum(t, SCAN_RANGE) + self.rng.normal(0, 0.005, t.shape)
        return t

    # interface ------------------------------------------------------------
    def reset(self) -> None:
        h = self.size / 2 - 0.3
        while True:
            self.x, self.y = self.rng.uniform(-h, h, 2)
            if self._clearance(self.x, self.y) > 0.4:
                break
        self.yaw = float(self.rng.uniform(-np.pi, np.pi))
        self.v = self.w = 0.0
        self.t = 0.0

    def reading(self) -> Reading:
        return Reading(self._scan(), self.v, self.w, self.x, self.y, self.t, yaw=self.yaw)

    def _drive(self, cmd: tuple[float, float], seconds: float, dt: float = 0.01) -> None:
        steps = int(round(seconds / dt))
        k = dt / self.lag
        for _ in range(steps):
            self.v += k * (cmd[0] - self.v)
            self.w += k * (cmd[1] - self.w)
            self.yaw += self.w * dt
            nx = self.x + self.v * np.cos(self.yaw) * dt
            ny = self.y + self.v * np.sin(self.yaw) * dt
            if self._clearance(nx, ny) < BODY_RADIUS:
                self.v = 0.0  # stopped by contact; turning in place stays possible
                self.bumps += 1
            else:
                self.x, self.y = nx, ny
            self.t += dt

    def tick(self, old, new, latency_s: float, period_s: float) -> Reading:
        self._drive(old, latency_s)
        self._drive(new, max(period_s - latency_s, 0.0))
        return self.reading()

    def close(self) -> None:
        pass


class Bridge:
    """Client for the ``limo_cadence_bridge`` node (Gazebo or the rover).

    Ticks are paced by the bridge's clock: simulation time in Gazebo, wall time on the rover.
    A sim running slower than real time therefore still sees exactly the emulated latency.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 20.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self.tick_start = 0.0  # bridge time of the last reading

    def _call(self, **msg) -> dict:
        self.file.write(json.dumps(msg) + "\n")
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise ConnectionError("bridge closed the connection")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError("bridge: " + reply["error"])
        return reply

    def now(self) -> float:
        return float(self._call(op="time")["t"])

    def _wait_until(self, t: float) -> None:
        while self.now() < t:
            time.sleep(0.002)

    def reading(self) -> Reading:
        r = self._call(op="read")
        self.tick_start = r["t"]
        return Reading(np.array(r["ranges"], float), r["v"], r["w"], r["x"], r["y"], r["t"],
                       r.get("tilt", 0.0), r.get("yaw", 0.0))

    def viz(self, **items) -> None:
        """Draw markers in the sim (goal, action, sectors); ignored where unsupported."""
        try:
            self._call(op="viz", **items)
        except RuntimeError:
            pass

    def tick(self, old, new, latency_s: float, period_s: float) -> Reading:
        # the bridge keeps republishing the old command while the brain "computes"
        self._wait_until(self.tick_start + latency_s)
        self._call(op="cmd", v=new[0], w=new[1])
        self._wait_until(self.tick_start + max(period_s, latency_s))
        return self.reading()

    def reset(self) -> None:
        self._call(op="cmd", v=0.0, w=0.0)
        self._call(op="reset")
        deadline = time.perf_counter() + 10.0
        while True:  # the bridge reports no odometry until the reset has settled
            try:
                self.reading()
                return
            except RuntimeError:
                if time.perf_counter() > deadline:
                    raise
                time.sleep(0.1)

    def close(self) -> None:
        try:
            self._call(op="cmd", v=0.0, w=0.0)
        finally:
            self.sock.close()

