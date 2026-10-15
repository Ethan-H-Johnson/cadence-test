"""Named overrides for the existing composed-brain configurations.

These are interface names and validation, not another set of founder values or
learning laws. The owning dataclasses continue to define their valid domains.
Mutation search spaces deliberately need not cover every valid configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, TypeVar

import numpy as np

from .arousal import ArousalConfig
from .learning import LearnerConfig
from .plasticity import ActorCriticConfig

Config = TypeVar("Config", LearnerConfig, ActorCriticConfig, ArousalConfig)

_KINDS = {
    "learning": LearnerConfig,
    "actor": ActorCriticConfig,
    "arousal": ArousalConfig,
}
_FIELDS = {prefix: {field.name for field in fields(kind)} for prefix, kind in _KINDS.items()}
_BOOLEAN = {"centered", "qualified", "surprise", "center_scale", "critic_normalize"}
_INTEGER = {"free_steps", "nudged_steps", "damping", "eligibility_steps", "youth"}
_STRING = {"nudge", "critic_signal"}
_OPTIONAL = {"eta_bias", "tolerance", "eligibility_steps"}


def real(name: str, value: Any, *, low: float = -np.inf, high: float = np.inf) -> float:
    """Accept real scalars, never truth values, arrays or coerced strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and lie in [{low:g}, {high:g}]")
    return number


def split_genes(genes: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Resolve only unambiguous public names; refuse typos before any changes."""
    values = dict(genes)
    if "temperature" in values:
        if "learning_temperature" in values:
            raise TypeError("use temperature or learning_temperature, not both")
        values["learning_temperature"] = values.pop("temperature")
    changes: dict[str, dict[str, Any]] = {prefix: {} for prefix in _KINDS}
    for name, value in values.items():
        if name in ("eta", "eta_bias"):
            raise TypeError(f"{name} is ambiguous; use actor_{name} or learning_{name}")
        prefix, _, field = name.partition("_")
        if prefix not in _FIELDS or field not in _FIELDS[prefix]:
            raise TypeError(f"unknown brain setting: {name!r}")
        changes[prefix][field] = value
    return tuple(changes[prefix] for prefix in _KINDS)


def configured(base: Config, changes: Mapping[str, Any], prefix: str) -> Config:
    """Overlay validated fields on a config without losing unspecified values."""
    expected = _KINDS[prefix]
    if not isinstance(base, expected):
        raise ValueError(f"{prefix} configuration must be {expected.__name__}")
    updates = dict(changes)
    unknown = set(updates) - _FIELDS[prefix]
    if unknown:
        raise TypeError(f"unknown {prefix} setting: {sorted(unknown)[0]!r}")
    # Config constructors validate ranges and relationships. Validate scalar kinds
    # here too: several legacy constructors accept bools as Python numbers.
    for name in _FIELDS[prefix]:
        value = updates.get(name, getattr(base, name))
        qualified = f"{prefix}_{name}"
        if value is None and name in _OPTIONAL and prefix != "arousal":
            continue
        if name in _BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"{qualified} must be boolean")
        elif name in _INTEGER:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{qualified} must be an integer")
            if name in updates:
                updates[name] = int(value)
        elif name in _STRING:
            if not isinstance(value, str):
                raise ValueError(f"{qualified} must be a string")
        else:
            number = real(qualified, value)
            if name in updates:
                updates[name] = number
    try:
        return replace(base, **updates) if updates else base
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {prefix} configuration: {exc}") from exc


def arousal_config(
    value: ArousalConfig | Mapping[str, Any] | bool | None,
    changes: Mapping[str, Any],
    *,
    current: ArousalConfig | None = None,
    retuning: bool = False,
) -> ArousalConfig | None:
    """At birth a mapping starts from founders; in life it patches current genes."""
    if value is None:
        if changes:
            raise ValueError("arousal settings require arousal=True or an arousal configuration")
        return None
    if isinstance(value, Mapping):
        overlap = set(value) & set(changes)
        if overlap:
            name = sorted(overlap)[0]
            raise TypeError(f"arousal_{name} was supplied both in arousal and as a named setting")
        return configured(current or ArousalConfig(), {**value, **changes}, "arousal")
    if value is True and not retuning:
        value = ArousalConfig()
    if not isinstance(value, ArousalConfig):
        raise ValueError(
            "arousal must be True, an ArousalConfig, a mapping, or None at construction"
        )
    return configured(value, changes, "arousal")


def plain(value: Any) -> Any:
    """Return detached, JSON-safe descriptive values, including NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(item) for item in value]
    return value
