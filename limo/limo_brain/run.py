"""Drive a LIMO with one continuing Cadence brain, optionally on an emulated small computer.

Examples (from the limo/ folder):

    python -m limo_brain.run --env toy --ticks 4000
    python -m limo_brain.run --env toy --profile orin-nano-super        # emulate the rover
    python -m limo_brain.run --env toy --profile orin-nano-super --calibration rover.json
    python -m limo_brain.run --env toy --weak-left 0.5 --weak-at 3000   # disturbance
    python -m limo_brain.run --env bridge --host 127.0.0.1 --minutes 10 # Gazebo or robot

--profile caps cores/threads and stretches each brain step's measured time by the target's
slowdown; the robot keeps executing the previous command for that long, and a step that
takes longer than the control period delays the next one, as on the slow computer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .budget import PROFILES, apply_threads, finish_budget, make_budget


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=("toy", "bridge"), default="toy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--ticks", type=int, default=4000, help="control ticks (toy)")
    ap.add_argument("--minutes", type=float, default=None, help="wall-clock limit (bridge)")
    ap.add_argument("--hz", type=float, default=10.0,
                    help="control rate; the rover's T-mini Pro lidar scans at 10 Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=("goal", "roam"), default="goal",
                    help="goal: reach a goal that moves when reached (4 more inputs); "
                         "roam: drive forward without bumping")
    ap.add_argument("--layout", choices=("open", "boxes"), default="open",
                    help="toy arena and goal sampling; match the Gazebo world (layout:=)")
    ap.add_argument("--policy", choices=("cadence", "reflex", "random"), default="cadence",
                    help="reflex and random are reference controllers that do not learn")
    ap.add_argument("--no-viz", action="store_true", help="do not draw markers in Gazebo")
    ap.add_argument("--need", type=float, default=None, help="arousal need of a new brain (default 0.3)")
    ap.add_argument("--stuck-reset", type=int, default=40,
                    help="respawn after this many consecutive bumped ticks (0 = never). A fall "
                         "(tilt > 45 deg) always ends the episode: respawn in sim, stop on the robot")
    g = ap.add_argument_group("resource budget")
    g.add_argument("--profile", choices=sorted(PROFILES), default="host")
    g.add_argument("--slowdown", type=float, help="override the profile's slowdown factor")
    g.add_argument("--cores", type=int, help="override the profile's core count")
    g.add_argument("--memory-mb", type=int, help="override the profile's memory allowance")
    g.add_argument("--hard-memory", action="store_true", help="enforce memory as a hard cap (Linux)")
    g.add_argument("--calibration", help="bench JSON measured on the real target")
    d = ap.add_argument_group("disturbance")
    d.add_argument("--weak-left", type=float, default=1.0, help="left-side wheel strength, e.g. 0.5")
    d.add_argument("--weak-at", type=int, default=0, help="tick at which the weakness starts")
    ap.add_argument("--load", help="continue a saved brain (.npz)")
    ap.add_argument("--save", help="save the brain at the end (.npz)")
    ap.add_argument("--window", type=int, default=500, help="ticks per printed summary row")
    ap.add_argument("--out", default="runs", help="folder for the run summary JSON")
    return ap.parse_args()


def main() -> None:
    a = parse()
    budget = make_budget(a.profile, slowdown=a.slowdown, cores=a.cores, memory_mb=a.memory_mb,
                         hard_memory=a.hard_memory, calibration=a.calibration)
    apply_threads(budget.cores)  # before NumPy loads

    import numpy as np

    from . import arena, robot
    from .bench import bench
    from .brain import NEED, new_brain
    from .envs import Bridge, ToySim
    from .viewer import Sender

    host_ms = None
    if budget.emulating and not budget.explicit_slowdown:
        host_ms = bench()["median_ms"]  # the same workload as a calibration file
    finish_budget(budget, host_ms)
    print("budget:", json.dumps({k: budget.report()[k] for k in ("profile", "slowdown", "slowdown_source", "cores")}))

    from cadence import Brain

    goal_task = a.task == "goal"
    inputs = robot.INPUTS + (robot.GOAL_INPUTS if goal_task else 0)
    brain = Brain.load(a.load) if a.load else new_brain(
        a.seed, need=NEED if a.need is None else a.need, inputs=inputs)
    env = ToySim(seed=a.seed, layout=a.layout) if a.env == "toy" else Bridge(a.host, a.port)
    rng = np.random.default_rng(a.seed)
    period = 1.0 / a.hz
    limit = None if a.minutes is None else time.monotonic() + 60 * a.minutes
    ticks = a.ticks if a.env == "toy" or a.minutes is None else 10**9

    rows, window, stuck, resets, reached_total = [], [], 0, 0, 0
    reading, previous, feedback = env.reading(), (0.0, 0.0), {}
    goal = arena.sample_goal(rng, (reading.x, reading.y), a.layout) if goal_task else None
    viz = a.env == "bridge" and not a.no_viz
    view = Sender()  # the native viewer (py -m limo_brain.viewer); harmless if none is open
    try:
        for t in range(ticks):
            if limit and time.monotonic() > limit:
                break
            start = time.perf_counter()
            obs = robot.observe_goal(reading, goal) if goal_task else robot.observe(reading)
            if a.policy == "cadence":
                action = int(brain.live(obs, **feedback)[0])
            elif a.policy == "reflex":
                action = robot.reflex_goal(obs) if goal_task else robot.reflex(obs)
            else:
                action = int(rng.integers(len(robot.ACTIONS)))
            measured = time.perf_counter() - start
            charged = budget.charge(measured, period)
            if viz and t % 3 == 0:  # ~3 Hz: the Gazebo window renders on the CPU by default
                name, v, w = robot.ACTIONS[action]
                env.viz(goal=goal, action=name, v=v, w=w,
                        sectors=[round(float(n), 3) for n in obs[0, :robot.SECTORS]])
            weak = a.weak_left if t >= a.weak_at else 1.0
            cmd = robot.command(action, weak)
            after = env.tick(previous, cmd, charged, period)
            reached = False
            if goal_task:
                pay, reached = robot.goal_reward(reading, after, goal)
                if reached:
                    reached_total += 1
                    goal = arena.sample_goal(rng, (after.x, after.y), a.layout)
            else:
                pay = robot.reward(reading, after)
            stuck = stuck + 1 if robot.bumped(after) else 0
            # a fall ends the episode at once; a long bump only with --stuck-reset
            done = robot.fallen(after) or (bool(a.stuck_reset) and stuck >= a.stuck_reset)
            feedback = {"reward": [pay], "done": [done]}
            aroused = a.policy == "cadence" and brain.last_arousal["mode"] == "aroused"
            window.append((pay, robot.bumped(after), aroused,
                           action, after.t - reading.t, charged > period, reached))
            name, v, w = robot.ACTIONS[action]
            view.send(tick=t + 1, x=after.x, y=after.y, yaw=after.yaw,
                      ranges=[round(float(r), 2) for r in after.ranges[::2]], goal=goal,
                      action=name, v=v, w=w, reward=pay, goals=reached_total, reset_count=resets,
                      mode=brain.last_arousal["mode"] if a.policy == "cadence" else "",
                      policy=a.policy, measured_ms=1000 * measured, charged_ms=1000 * charged,
                      env=a.env, layout=a.layout)
            reading, previous = after, cmd
            if done:
                if robot.fallen(after) and a.env == "bridge" and not a.stuck_reset:
                    print("robot fell over; stopping (no respawn on a real robot)")
                    break
                env.reset()
                reading, previous, stuck, resets = env.reading(), (0.0, 0.0), 0, resets + 1
            if len(window) == a.window:
                rows.append(summarise(t + 1, window, weak))
                print(json.dumps(rows[-1]))
                window = []
    except KeyboardInterrupt:
        print("stopped")
    finally:
        env.close()
    if window:
        rows.append(summarise(t + 1, window, weak))
        print(json.dumps(rows[-1]))
    if a.save and a.policy == "cadence":
        brain.save(a.save)
    report = {"args": vars(a), "budget": budget.report(), "resets": resets,
              "goals_reached": reached_total, "windows": rows}
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    path = out / time.strftime("run-%Y%m%d-%H%M%S.json")
    path.write_text(json.dumps(report, indent=2))
    print("resets:", resets, "goals reached:", reached_total, "budget:", json.dumps(budget.report()))
    print("summary written to", path)


def summarise(tick: int, window: list, weak: float) -> dict:
    pay, bump, aroused, action, dt, over, reached = zip(*window)
    counts = [action.count(i) for i in range(5)]
    return {
        "tick": tick,
        "mean_reward": round(sum(pay) / len(pay), 3),
        "bump_share": round(sum(bump) / len(bump), 3),
        "aroused_share": round(sum(aroused) / len(aroused), 3),
        "forward_share": round(counts[0] / len(action), 3),
        "actions": counts,
        "mean_tick_s": round(sum(dt) / len(dt), 4),
        "overrun_share": round(sum(over) / len(over), 3),
        "goals_reached": sum(reached),
        "weak_left": weak,
    }


if __name__ == "__main__":
    main()
