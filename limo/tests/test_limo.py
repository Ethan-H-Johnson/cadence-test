"""Smoke tests for the LIMO controller: run with `py -m pytest limo/tests` from the repo root."""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limo_brain import robot  # noqa: E402
from limo_brain.budget import REFERENCE_HOST_MS, finish_budget, make_budget  # noqa: E402
from limo_brain.envs import Bridge, ToySim  # noqa: E402


def test_profile_slowdown_rescales_by_host_speed():
    b = finish_budget(make_budget("jetson-nano"), host_bench_ms=REFERENCE_HOST_MS / 2)
    assert b.slowdown == pytest.approx(20.0)  # a host twice as fast needs twice the stretch
    assert b.cores == 4


def test_calibration_file_sets_slowdown(tmp_path):
    cal = tmp_path / "nano.json"
    cal.write_text(json.dumps({"median_ms": 30.0}))
    b = finish_budget(make_budget("jetson-nano", calibration=str(cal)), host_bench_ms=3.0)
    assert b.slowdown == pytest.approx(10.0)
    assert b.source.startswith("calibration")


def test_explicit_slowdown_wins_and_overruns_count():
    b = finish_budget(make_budget("pi4", slowdown=50.0), host_bench_ms=1.0)
    assert b.slowdown == 50.0
    assert b.charge(0.004, period_s=0.125) == pytest.approx(0.2)
    assert b.overruns == 1


def test_weak_left_turns_left():
    # a slower left side turns the rover left (positive yaw) and loses a quarter of its speed
    v, w = robot.command(0, weak_left=0.5)
    assert v == pytest.approx(0.75 * robot.FAST)
    assert w > 0


def test_toy_latency_delays_new_command():
    sim = ToySim(seed=1, obstacles=0)
    sim.reset()
    sim.x = sim.y = 0.0
    fast = sim.tick((0.0, 0.0), (0.3, 0.0), latency_s=0.0, period_s=0.5)
    sim.reset()
    sim.x = sim.y = 0.0
    sim.t = 0.0
    slow = sim.tick((0.0, 0.0), (0.3, 0.0), latency_s=0.4, period_s=0.5)
    assert fast.v > slow.v  # the late command has had less time to take effect


def test_bridge_client_against_stub_server():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    commands = []

    def serve():
        conn, _ = server.accept()
        f = conn.makefile("rwb")
        for line in iter(f.readline, b""):
            req = json.loads(line)
            if req["op"] == "read":
                reply = {"ranges": [1.0] * 240, "v": 0.1, "w": 0.0, "x": 0.0, "y": 0.0,
                         "t": time.monotonic(), "tilt": 0.0}
            elif req["op"] == "time":
                reply = {"t": time.monotonic()}
            else:
                commands.append(req)
                reply = {"ok": True}
            f.write((json.dumps(reply) + "\n").encode())
            f.flush()
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    env = Bridge("127.0.0.1", port)
    r = env.reading()
    assert robot.observe(r).shape == (1, robot.INPUTS)
    env.tick((0, 0), (0.3, 0.0), latency_s=0.0, period_s=0.01)
    env.close()
    assert commands[0] == {"op": "cmd", "v": 0.3, "w": 0.0}
    assert commands[-1] == {"op": "cmd", "v": 0.0, "w": 0.0}  # close stops the base


def test_reflex_beats_random_in_toy():
    def run(policy):
        sim, rng, pays = ToySim(seed=2), np.random.default_rng(0), []
        r, prev = sim.reading(), (0.0, 0.0)
        for _ in range(400):
            obs = robot.observe(r)
            a = robot.reflex(obs) if policy == "reflex" else int(rng.integers(5))
            cmd = robot.command(a)
            after = sim.tick(prev, cmd, 0.002, 0.125)
            pays.append(robot.reward(r, after))
            r, prev = after, cmd
        return np.mean(pays)

    assert run("reflex") > run("random") + 0.2
