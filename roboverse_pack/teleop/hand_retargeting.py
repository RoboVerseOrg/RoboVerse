from __future__ import annotations

"""Optional Wuji hand retargeting bridge for teleoperation landmarks."""

import os
import sys
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np


class _RetargeterProtocol(Protocol):
    def reset(self) -> None: ...
    def retarget_left(self, fingers: np.ndarray) -> np.ndarray: ...
    def retarget_right(self, fingers: np.ndarray) -> np.ndarray: ...


_DEFAULT_OPENARM_TELEOP_ROOT = Path(
    os.environ.get(
        "ROBOVERSE_OPENARM_TELEOP_ROOT",
        os.environ.get("BIDEXBENCH_OPENARM_TELEOP_ROOT", Path(__file__).resolve().parent / "configs"),
    )
).expanduser()
DEFAULT_WUJI_HAND_RETARGETING_CONFIG = (
    _DEFAULT_OPENARM_TELEOP_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_avp.yaml"
).resolve()


def coerce_finger_landmarks(fingers) -> np.ndarray | None:
    """Return sanitized 21x3 finger landmarks, or None when unavailable."""
    if fingers is None:
        return None
    try:
        landmarks = np.asarray(fingers, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if landmarks.shape != (21, 3):
        return None
    return np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


class WujiHandRetargetingRuntime:
    """Lazy Wuji hand retargeting runtime with optional dependency loading."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        retargeter_factory=None,
        warn_unavailable: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_WUJI_HAND_RETARGETING_CONFIG
        self._retargeter_factory = retargeter_factory
        self._warn_unavailable = bool(warn_unavailable)
        self._retargeter: _RetargeterProtocol | None = None
        self._load_error: Exception | None = None
        self._warned_unavailable = False

    def _build_default_retargeter(self, config_path: Path) -> _RetargeterProtocol:
        openarm_teleop_root = str(config_path.parents[2])
        if openarm_teleop_root not in sys.path:
            sys.path.insert(0, openarm_teleop_root)
        from wuji_retargeting import Retargeter

        class _DirectHandRetargeter:
            def __init__(self, path: str) -> None:
                self._left = Retargeter.from_yaml(path, hand_side="left")
                self._right = Retargeter.from_yaml(path, hand_side="right")

            def reset(self) -> None:
                self._left.reset()
                self._right.reset()

            def retarget_left(self, fingers: np.ndarray) -> np.ndarray:
                return self._left.retarget(fingers)

            def retarget_right(self, fingers: np.ndarray) -> np.ndarray:
                return self._right.retarget(fingers)

        return _DirectHandRetargeter(str(config_path))

    def _ensure_loaded(self) -> None:
        if self._retargeter is not None or self._load_error is not None:
            return
        try:
            factory = self._retargeter_factory or self._build_default_retargeter
            self._retargeter = factory(self.config_path)
        except Exception as exc:  # pragma: no cover - availability path depends on optional external retargeter
            self._load_error = exc

    def reset(self) -> None:
        """Reset the underlying retargeter if it is available."""
        self._ensure_loaded()
        if self._retargeter is not None:
            self._retargeter.reset()

    def retarget(
        self,
        hand_key: str,
        fingers,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Retarget 21x3 hand landmarks to a 20-joint Wuji target."""
        landmarks = coerce_finger_landmarks(fingers)
        if landmarks is None or np.allclose(landmarks, 0.0):
            return None

        self._ensure_loaded()
        if self._retargeter is None:
            self._maybe_warn_unavailable()
            return None

        if hand_key == "left":
            joint_targets = self._retargeter.retarget_left(landmarks)
        elif hand_key == "right":
            joint_targets = self._retargeter.retarget_right(landmarks)
        else:
            raise ValueError(f"invalid hand key: {hand_key!r}")

        joint_targets = np.asarray(joint_targets, dtype=np.float32).reshape(-1)
        if joint_targets.shape != (20,):
            return None
        if joint_lower is not None and joint_upper is not None:
            joint_targets = np.clip(joint_targets, joint_lower, joint_upper)
        return joint_targets.astype(np.float32, copy=False)

    def _maybe_warn_unavailable(self) -> None:
        if not self._warn_unavailable or self._warned_unavailable:
            return
        self._warned_unavailable = True
        error = repr(self._load_error) if self._load_error is not None else "unknown"
        warnings.warn(
            f"hand_retargeting_unavailable config={self.config_path} error={error}",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = [
    "DEFAULT_WUJI_HAND_RETARGETING_CONFIG",
    "WujiHandRetargetingRuntime",
    "coerce_finger_landmarks",
]
