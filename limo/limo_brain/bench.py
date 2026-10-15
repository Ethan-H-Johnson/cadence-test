"""Time one controller brain step on this computer.

Run it on the real target to replace a profile's estimated slowdown with a measurement:

    python3 -m limo_brain.bench --save rover.json        # on the rover (Orin Nano Super)
    py -m limo_brain.run --profile orin-nano-super --calibration rover.json  # on the dev PC

The workload is the controller's own brain living on fixed random readings and rewards,
so the same steps are timed everywhere.
"""

from __future__ import annotations

import argparse
import json
import platform
import time


def bench(steps: int = 200, warmup: int = 30, seed: int = 0, modules=None) -> dict:
    import numpy as np

    from .brain import MODULES, new_brain
    from .robot import INPUTS

    brain = new_brain(seed, modules or MODULES)
    rng = np.random.default_rng(seed)
    obs = rng.random((warmup + steps, 1, INPUTS))
    pay = rng.uniform(-1, 1, warmup + steps)
    times = []
    for i in range(warmup + steps):
        feedback = {} if i == 0 else {"reward": [pay[i]], "done": [False]}
        t = time.perf_counter()
        brain.live(obs[i], **feedback)
        if i >= warmup:
            times.append(time.perf_counter() - t)
    ms = np.array(times) * 1000
    return {
        "median_ms": round(float(np.median(ms)), 4),
        "p95_ms": round(float(np.percentile(ms, 95)), 4),
        "max_ms": round(float(ms.max()), 4),
        "steps": steps,
        "modules": list(modules or MODULES),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cores", type=int, default=1, help="math-library threads (default 1)")
    ap.add_argument("--save", help="write the result as JSON (a calibration file)")
    a = ap.parse_args()
    from .budget import apply_threads

    apply_threads(a.cores)
    result = bench(a.steps)
    print(json.dumps(result, indent=2))
    if a.save:
        with open(a.save, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
