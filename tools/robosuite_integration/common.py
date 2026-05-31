"""Shared types and helpers for the robosuite ↔ MetaSim parity harness.

The harness compares physics-engine evolution of *the same* compiled robosuite
MJCF stepped two ways:

* ``robosuite_native`` — robosuite's own ``MjSim`` (native ``mujoco`` binding).
* ``metasim_dm_control`` — the dm_control / PyMJCF round-trip that MetaSim's
  ``MujocoHandler`` uses to compose models.

Both ultimately step ``mujoco.mj_step`` on a ``mujoco.MjModel``; the question is
whether MetaSim's *loader* preserves the model bit-for-bit. We drive the raw
actuators with a deterministic, controller-free signal so the comparison
isolates the engine from robosuite's OSC controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# A control function maps (control-step index, qpos, qvel) -> raw actuator ctrl.
CtrlFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class RolloutResult:
    """Time series captured from one controller-free engine rollout."""

    backend: str  # "robosuite_native" | "metasim_dm_control" | "raw_mujoco"
    task_id: str
    dt: float  # base physics timestep (s)
    decimation: int  # physics substeps per control step
    qpos: np.ndarray  # (T+1, nq) including the initial frame
    qvel: np.ndarray  # (T+1, nv)
    ctrl: np.ndarray  # (T, nu) applied each control step
    time: np.ndarray  # (T+1,)
    joint_names: list[str]
    actuator_names: list[str]

    def summary(self) -> dict[str, float]:
        return {
            "backend": self.backend,
            "task_id": self.task_id,
            "T": int(self.qpos.shape[0] - 1),
            "nq": int(self.qpos.shape[1]),
            "nv": int(self.qvel.shape[1]),
            "nu": int(self.ctrl.shape[1]),
            "dt": float(self.dt),
            "decimation": int(self.decimation),
            "max_abs_qpos": float(np.abs(self.qpos).max()),
            "max_abs_qvel": float(np.abs(self.qvel).max()),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            backend=self.backend,
            task_id=self.task_id,
            dt=self.dt,
            decimation=self.decimation,
            qpos=self.qpos,
            qvel=self.qvel,
            ctrl=self.ctrl,
            time=self.time,
            joint_names=np.array(self.joint_names, dtype=object),
            actuator_names=np.array(self.actuator_names, dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RolloutResult:
        z = np.load(path, allow_pickle=True)
        return cls(
            backend=str(z["backend"]),
            task_id=str(z["task_id"]),
            dt=float(z["dt"]),
            decimation=int(z["decimation"]),
            qpos=z["qpos"],
            qvel=z["qvel"],
            ctrl=z["ctrl"],
            time=z["time"],
            joint_names=list(z["joint_names"]),
            actuator_names=list(z["actuator_names"]),
        )


def zero_ctrl(nu: int) -> CtrlFn:
    """Apply zero actuator command every step (passive dynamics under gravity)."""

    def _fn(t: int, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return np.zeros(nu, dtype=np.float64)

    return _fn


def sinusoid_ctrl(nu: int, *, amp: float = 0.4, period: int = 40) -> CtrlFn:
    """Drive every actuator with a phase-shifted sinusoid — exercises all DOFs.

    This is a stronger engine-parity stressor than gravity-only: it pushes the
    arm through large excursions and contacts, amplifying any divergence.
    """

    def _fn(t: int, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * t / period
        return amp * np.sin(phase + np.arange(nu) * 0.7)

    return _fn


CTRL_FNS: dict[str, Callable[[int], CtrlFn]] = {
    "zero": zero_ctrl,
    "sinusoid": sinusoid_ctrl,
}
