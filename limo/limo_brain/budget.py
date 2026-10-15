"""Emulate a smaller computer's resources on the development host.

A budget does three things:

* caps cores and math-library threads (applied before NumPy is imported),
* stretches the measured time of each brain step by a slowdown factor, so the
  robot (simulated or real) sees the latency the target computer would add,
* records peak memory against the target's allowance (a hard cap on Linux with
  ``hard_memory``).

Slowdown factors in ``PROFILES`` are *estimates* relative to the reference host
(an i9-9900K desktop, whose bench median is ``REFERENCE_HOST_MS``). They are
rescaled by this host's own bench, so a faster or slower development machine
still emulates the same target. Replace an estimate with a measurement by
running ``python -m limo_brain.bench --save nano.json`` on the real device and
passing ``--calibration nano.json``.

Only stdlib is imported here: ``apply_threads`` must run before NumPy loads.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Bench median (ms per live step of the default controller brain) on the reference host.
# Measured with `python -m limo_brain.bench` on an i9-9900K, Windows 11, Python 3.14.
REFERENCE_HOST_MS = 2.16


@dataclass(frozen=True)
class Profile:
    name: str
    slowdown: float  # estimated brain-step time on the target / on the reference host
    cores: int | None
    memory_mb: int | None
    note: str


PROFILES = {
    "host": Profile("host", 1.0, None, None, "no emulation"),
    # the rover: Jetson Orin Nano Super (6x Cortex-A78AE @1.7 GHz, 8 GB shared with the GPU).
    # Measured 2026-10-15: bench median 13.25 ms on the rover (Python 3.12.3, NumPy 1.26.4)
    # against 2.16 ms here, so 6.1x. limo/rover.json holds the measurement.
    "orin-nano-super": Profile(
        "orin-nano-super", 6.1, 6, 3000,
        "6x Cortex-A78AE @1.7 GHz, 8 GB shared with GPU, ROS 2 and the OS; slowdown measured",
    ),
    "jetson-nano": Profile(
        "jetson-nano", 10.0, 4, 1500,
        "4x Cortex-A57 @1.43 GHz, 4 GB shared with GPU, ROS and the OS; slowdown estimated",
    ),
    "pi4": Profile("pi4", 7.0, 4, 1500, "4x Cortex-A72 @1.8 GHz; slowdown estimated"),
    "pi5": Profile("pi5", 3.0, 4, 2000, "4x Cortex-A76 @2.4 GHz; slowdown estimated"),
}


@dataclass
class Budget:
    profile: Profile
    slowdown: float = 1.0
    cores: int | None = None
    memory_mb: int | None = None
    hard_memory: bool = False
    source: str = "none"
    calibration: str | None = None
    explicit_slowdown: bool = False
    # running tallies
    steps: int = 0
    overruns: int = 0
    measured_s: float = 0.0
    charged_s: float = 0.0
    worst_charged_s: float = 0.0
    _notes: list[str] = field(default_factory=list)

    @property
    def emulating(self) -> bool:
        return self.profile.name != "host" or self.slowdown != 1.0

    def charge(self, measured_s: float, period_s: float) -> float:
        """The time this step would have taken on the target; tallies overruns of the period."""
        charged = measured_s * self.slowdown
        self.steps += 1
        self.measured_s += measured_s
        self.charged_s += charged
        self.worst_charged_s = max(self.worst_charged_s, charged)
        if charged > period_s:
            self.overruns += 1
        return charged

    def report(self) -> dict:
        n = max(self.steps, 1)
        peak = peak_rss_mb()
        out = {
            "profile": self.profile.name,
            "slowdown": round(self.slowdown, 2),
            "slowdown_source": self.source,
            "cores": self.cores,
            "steps": self.steps,
            "mean_measured_ms": round(1000 * self.measured_s / n, 3),
            "mean_charged_ms": round(1000 * self.charged_s / n, 3),
            "worst_charged_ms": round(1000 * self.worst_charged_s, 3),
            "period_overruns": self.overruns,
            "peak_rss_mb": None if peak is None else round(peak, 1),
            "memory_budget_mb": self.memory_mb,
            "notes": self._notes,
        }
        if peak is not None and self.memory_mb is not None:
            out["memory_within_budget"] = peak <= self.memory_mb
        return out


def apply_threads(cores: int | None) -> None:
    """Pin math-library threads and CPU affinity. Call before importing NumPy."""
    threads = str(cores or 1) if cores else None
    if threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = threads
    if not cores:
        return
    mask_cpus = list(range(min(cores, os.cpu_count() or cores)))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, mask_cpus)
    elif sys.platform == "win32":
        import ctypes

        mask = sum(1 << c for c in mask_cpus)
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask))


def apply_memory_cap(memory_mb: int) -> bool:
    """Hard address-space cap; Linux/macOS only. Returns whether it was applied."""
    try:
        import resource
    except ImportError:
        return False
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return True


def peak_rss_mb() -> float | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**20
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024


def make_budget(
    profile: str = "host",
    *,
    slowdown: float | None = None,
    cores: int | None = None,
    memory_mb: int | None = None,
    hard_memory: bool = False,
    calibration: str | None = None,
) -> Budget:
    """Resolve a profile and overrides. Call ``apply_threads(budget.cores)`` before NumPy loads,
    then ``finish_budget`` once NumPy and Cadence are available."""
    p = PROFILES[profile]
    b = Budget(
        profile=p,
        cores=cores if cores is not None else p.cores,
        memory_mb=memory_mb if memory_mb is not None else p.memory_mb,
        hard_memory=hard_memory,
        calibration=calibration,
        explicit_slowdown=slowdown is not None,
    )
    if slowdown is not None:
        b.slowdown, b.source = slowdown, "--slowdown"
    elif calibration is not None:
        b.source = f"calibration:{calibration}"
    elif p.name != "host":
        b.source = "estimate"
    return b


def finish_budget(b: Budget, host_bench_ms: float | None) -> Budget:
    """Rescale the slowdown by this host's bench, apply the memory cap. Needs the brain bench."""
    if b.explicit_slowdown:
        pass
    elif b.calibration:
        target = json.loads(Path(b.calibration).read_text())["median_ms"]
        b._notes.append("calibration median %.3f ms" % target)
        b.slowdown = target / host_bench_ms if host_bench_ms else 1.0
    elif b.profile.name != "host" and host_bench_ms:
        b.slowdown = b.profile.slowdown * REFERENCE_HOST_MS / host_bench_ms
    if host_bench_ms:
        b._notes.append("host bench median %.3f ms" % host_bench_ms)
    if b.slowdown < 1.0:
        b._notes.append("target is faster than this host; no slowdown applied")
        b.slowdown = 1.0
    if b.hard_memory and b.memory_mb:
        if not apply_memory_cap(b.memory_mb):
            b._notes.append("hard memory cap unsupported on this OS; peak memory is reported instead")
    return b

