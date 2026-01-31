from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalRobotCommand:
    """Vendor-agnostic, joint-indexed low-level command.

    Arrays are aligned with ``joint_names``. Values can be None if the protocol does
    not provide that field.
    """

    joint_names: list[str]
    q_des: np.ndarray | None = None
    qd_des: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    tau_ff: np.ndarray | None = None
    mode: np.ndarray | None = None


@dataclass(frozen=True)
class SimRobotObservation:
    """Canonical observation extracted from a simulator handler (single env)."""

    joint_names_sorted: list[str]
    q_sorted: np.ndarray  # (J,)
    dq_sorted: np.ndarray  # (J,)
    tau_sorted: np.ndarray | None  # (J,) if available

    root_state: np.ndarray  # (13,) pos(3), quat_wxyz(4), lin_vel_world(3), ang_vel_world(3)
    body_names_sorted: list[str] | None
    body_state: np.ndarray | None  # (B,13) or None
