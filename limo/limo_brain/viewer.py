"""Live top-down view of a run: arena, robot, lidar, goal, the brain's choice and stats.

A native Windows window (tkinter), independent of Gazebo's GUI. ``limo_brain.run`` sends one
small UDP packet per tick to 127.0.0.1:5600; start the viewer before or after the run:

    py -m limo_brain.viewer                 # from limo/
    py -m limo_brain.viewer --layout boxes  # draw the boxed arena

Works for the toy sim, Gazebo and the rover alike (poses are in the arena / odometry frame).
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import tkinter as tk

from . import arena

PORT = 5600
SCALE = 120  # pixels per metre
MARGIN = 30
ACTION_COLOURS = {"forward": "#3b82f6", "veer left": "#f59e0b", "veer right": "#f59e0b",
                  "spin left": "#a855f7", "spin right": "#a855f7"}


class Viewer:
    def __init__(self, layout: str, port: int):
        self.layout = layout
        self.size = int(2 * arena.HALF * SCALE + 2 * MARGIN)
        self.root = tk.Tk()
        self.root.title("Cadence LIMO viewer")
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size + 70, bg="#f8fafc",
                                highlightthickness=0)
        self.canvas.pack()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.trail: list[tuple[float, float]] = []
        self.state: dict | None = None
        self.root.after(30, self.poll)

    def px(self, x: float, y: float) -> tuple[float, float]:
        """Arena metres to canvas pixels (x right, y up)."""
        return MARGIN + (x + arena.HALF) * SCALE, MARGIN + (arena.HALF - y) * SCALE

    def poll(self) -> None:
        latest = None
        while True:  # drain the socket, keep only the newest packet
            try:
                latest = self.sock.recv(65536)
            except BlockingIOError:
                break
        if latest:
            state = json.loads(latest)
            if self.state and state.get("reset_count") != self.state.get("reset_count"):
                self.trail.clear()
            self.state = state
            self.trail.append((state["x"], state["y"]))
            self.trail = self.trail[-400:]
            self.draw()
        self.root.after(30, self.poll)

    def draw(self) -> None:
        c, s = self.canvas, self.state
        c.delete("all")
        layout = s.get("layout", self.layout)
        # arena
        for (x0, y0), (x1, y1) in arena.SEGMENTS[layout]:
            c.create_line(*self.px(x0, y0), *self.px(x1, y1), width=3, fill="#475569")
        # goal and its arrival zone
        if s.get("goal"):
            gx, gy = s["goal"]
            r = 0.3 * SCALE
            px, py = self.px(gx, gy)
            c.create_oval(px - r, py - r, px + r, py + r, outline="#16a34a", width=2, fill="#dcfce7")
            c.create_oval(px - 6, py - 6, px + 6, py + 6, fill="#16a34a", outline="")
        # trail
        if len(self.trail) > 1:
            c.create_line(*[v for p in self.trail for v in self.px(*p)], fill="#cbd5e1", width=2)
        x, y, yaw = s["x"], s["y"], s["yaw"]
        # lidar points (front 240 deg, right to left)
        ranges = s.get("ranges") or []
        lx, ly = x + 0.103 * math.cos(yaw), y + 0.103 * math.sin(yaw)
        for i, rng in enumerate(ranges):
            if rng >= 11.9:
                continue
            a = yaw + math.radians(-120 + 240 * i / max(len(ranges) - 1, 1))
            px, py = self.px(lx + rng * math.cos(a), ly + rng * math.sin(a))
            c.create_rectangle(px - 1, py - 1, px + 1, py + 1, outline="#ef4444", fill="#ef4444")
        # predicted path of the chosen command over 1.5 s
        v, w = s.get("v", 0.0), s.get("w", 0.0)
        colour = ACTION_COLOURS.get(s.get("action"), "#0f172a")
        pts, px_, py_, pyaw = [], x, y, yaw
        for _ in range(16):
            pts += self.px(px_, py_)
            pyaw += w * 0.1
            px_ += max(v, 0.08) * math.cos(pyaw) * 0.1
            py_ += max(v, 0.08) * math.sin(pyaw) * 0.1
        c.create_line(*pts, fill=colour, width=3, arrow=tk.LAST)
        # robot: 0.32 x 0.22 m body with a nose mark
        corners = [(0.16, 0.11), (0.16, -0.11), (-0.16, -0.11), (-0.16, 0.11)]
        body = [v for dx, dy in corners for v in self.px(x + dx * math.cos(yaw) - dy * math.sin(yaw),
                                                          y + dx * math.sin(yaw) + dy * math.cos(yaw))]
        c.create_polygon(*body, fill="#1e293b", outline="#0f172a")
        c.create_line(*self.px(x, y), *self.px(x + 0.16 * math.cos(yaw), y + 0.16 * math.sin(yaw)),
                      fill="#facc15", width=3)
        # stats
        top = self.size + 8
        line1 = "tick {tick}   {action}   reward {reward:+.2f}   goals {goals}   resets {resets}".format(
            tick=s.get("tick", 0), action=s.get("action", "?"), reward=s.get("reward", 0.0),
            goals=s.get("goals", 0), resets=s.get("reset_count", 0))
        line2 = "{mode}   {policy}   brain step {ms:.1f} ms (emulated {charged:.1f} ms)   {env}".format(
            mode=s.get("mode", ""), policy=s.get("policy", ""), ms=s.get("measured_ms", 0.0),
            charged=s.get("charged_ms", 0.0), env=s.get("env", ""))
        c.create_text(MARGIN, top, anchor="nw", text=line1, font=("Segoe UI", 12, "bold"), fill="#0f172a")
        c.create_text(MARGIN, top + 26, anchor="nw", text=line2, font=("Segoe UI", 11), fill="#334155")

    def run(self) -> None:
        self.root.mainloop()


class Sender:
    """Used by ``limo_brain.run``: fire-and-forget UDP, so a missing viewer costs nothing."""

    def __init__(self, port: int = PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", port)

    def send(self, **state) -> None:
        try:
            self.sock.sendto(json.dumps(state).encode(), self.addr)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=arena.LAYOUTS, default="open")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    Viewer(a.layout, a.port).run()


if __name__ == "__main__":
    main()
