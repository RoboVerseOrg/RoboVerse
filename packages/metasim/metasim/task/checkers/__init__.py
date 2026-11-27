"""Success predicates over the simulator state: checkers and the detectors they use.

A checker answers "is the task solved in each env *now*" from a :class:`~metasim.types.TensorState`
(``check(handler, states) -> (num_envs,) bool``) and keeps whatever per-env reference it needs
(an initial pose, a placed region), refreshed by ``reset(handler, env_ids)``. A task returns
``checker.check(...)`` from ``_terminated`` (or ``_success``) and resets it in ``_on_reset``. How
many steps the predicate must hold, or whether the first hit counts, is the evaluation protocol's
call (``metasim.bench.EvalProtocol``), not the checker's.
"""

from __future__ import annotations

from .checkers import (
    DetectedChecker,
    EmptyChecker,
    JointPosChecker,
    JointPosShiftChecker,
    PositionShiftChecker,
    RotationShiftChecker,
)
from .detectors import (
    BaseDetector,
    CombinedDetector,
    Relative2DSphereDetector,
    Relative3DSphereDetector,
    RelativeBboxDetector,
)

__all__ = [
    "BaseDetector",
    "CombinedDetector",
    "DetectedChecker",
    "EmptyChecker",
    "JointPosChecker",
    "JointPosShiftChecker",
    "PositionShiftChecker",
    "Relative2DSphereDetector",
    "Relative3DSphereDetector",
    "RelativeBboxDetector",
    "RotationShiftChecker",
]
