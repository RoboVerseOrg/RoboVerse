"""Boot-time checks for scenario configs: reject an invalid value where it is written, with a fix.

A bad ``num_envs`` / ``dt`` / camera size otherwise surfaces minutes later as a backend-specific
error (a MuJoCo compile failure, a CUDA index error, a zero-sized render buffer) that names no
config field. Each check says which field, what it got, and what it accepts. Values are checked, not
retyped: a numpy / torch scalar that behaves like the right number is accepted and stored as given.

The checks run at construction and in ``ScenarioCfg.update``; ``from_dict`` and direct attribute
assignment bypass them.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Any


def _reject(owner: str, field: str, value: Any, expected: str) -> None:
    raise ValueError(f"{owner}.{field}={value!r} is invalid: expected {expected}.")


def _as_int(value: Any) -> int | None:
    """``int`` for anything that is an exact integer (numpy ints, 0-d integer tensors); None otherwise."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, numbers.Integral):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    """``float`` for real numbers and 0-d numeric tensors; None for bools, strings and everything else."""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    item = getattr(value, "item", None)  # numpy / torch scalars
    if callable(item) and getattr(value, "shape", None) == ():
        inner = item()
        return float(inner) if isinstance(inner, numbers.Real) and not isinstance(inner, bool) else None
    return None


def positive_int(owner: str, field: str, value: Any) -> None:
    """An integer >= 1 (``bool`` is not one)."""
    as_int = _as_int(value)
    if as_int is None or as_int < 1:
        _reject(owner, field, value, "an integer >= 1")


def positive_finite_or_none(owner: str, field: str, value: Any) -> None:
    """``None`` (backend default) or a finite number > 0; ``""``, ``nan`` and ``inf`` are rejected."""
    if value is None:
        return
    as_float = _as_float(value)
    if as_float is None or not math.isfinite(as_float) or as_float <= 0:
        _reject(owner, field, value, "None or a finite number > 0")


def finite_triple(owner: str, field: str, value: Any) -> None:
    """Three finite numbers (a position, a look-at point, a colour)."""
    try:
        items = [_as_float(v) for v in value]
    except TypeError:
        items = []
    if len(items) != 3 or any(v is None or not math.isfinite(v) for v in items):
        _reject(owner, field, value, "three finite numbers")


def sequence_of_configs(owner: str, field: str, value: Any) -> list:
    """A list (a tuple is accepted and converted); a bare config is the classic mistake."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError(
        f"{owner}.{field}={value!r} is invalid: expected a list of configs (wrap a single config in [...])."
    )
