"""The controller brain: one continuing Cadence life for the rover."""

from __future__ import annotations

import sys
from pathlib import Path

# Use this checkout's Cadence (the fork) unless cadence-net is installed.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cadence import ArousalConfig, Brain  # noqa: E402

from .robot import ACTIONS, INPUTS  # noqa: E402

MODULES = (48, 24)
# The reward per tick the body requires. Cruising forward pays ~0.5. With the founder's
# need of 0, chronic pain (pushing into a wall) becomes the usual level and the brain calms.
NEED = 0.3


def new_brain(seed: int = 0, modules: tuple[int, ...] = MODULES, need: float = NEED,
              inputs: int = INPUTS) -> Brain:
    return Brain.compose(
        inputs, len(ACTIONS), modules=modules, seed=seed, arousal=ArousalConfig(need=need)
    )
