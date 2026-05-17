"""Compare two `RolloutResult` traces and quantify divergence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .raw_rollout import RolloutResult


@dataclass
class DiffStats:
    a_backend: str
    b_backend: str
    task_id: str
    nq: int
    nv: int
    T: int
    qpos_l2: float  # final-step L2 distance over all DOFs
    qpos_max_abs: float  # max abs(diff) anywhere in trace
    qpos_rms: float  # RMS over all (t, dof)
    qvel_l2: float
    qvel_max_abs: float
    qvel_rms: float
    matches_bitwise: bool  # qpos identical to last bit
    matches_1e_8: bool
    matches_1e_5: bool
    matches_1e_2: bool

    def as_row(self) -> dict:
        return {
            "task": self.task_id,
            "a": self.a_backend,
            "b": self.b_backend,
            "T": self.T,
            "nq": self.nq,
            "nv": self.nv,
            "qpos_l2_final": self.qpos_l2,
            "qpos_max_abs": self.qpos_max_abs,
            "qpos_rms": self.qpos_rms,
            "qvel_max_abs": self.qvel_max_abs,
            "qvel_rms": self.qvel_rms,
            "bitwise": self.matches_bitwise,
            "match_1e-8": self.matches_1e_8,
            "match_1e-5": self.matches_1e_5,
            "match_1e-2": self.matches_1e_2,
        }


def diff_rollouts(a: RolloutResult, b: RolloutResult) -> DiffStats:
    """Compare two trajectories.

    Assumes both rollouts share the same T, nq, nv, control sequence.
    """
    if a.qpos.shape != b.qpos.shape:
        raise ValueError(f"qpos shape mismatch: {a.qpos.shape} vs {b.qpos.shape}")
    if a.qvel.shape != b.qvel.shape:
        raise ValueError(f"qvel shape mismatch: {a.qvel.shape} vs {b.qvel.shape}")

    dq = a.qpos - b.qpos
    dv = a.qvel - b.qvel

    return DiffStats(
        a_backend=a.backend,
        b_backend=b.backend,
        task_id=a.task_id,
        nq=a.qpos.shape[1],
        nv=a.qvel.shape[1],
        T=a.qpos.shape[0] - 1,
        qpos_l2=float(np.linalg.norm(dq[-1])),
        qpos_max_abs=float(np.abs(dq).max()),
        qpos_rms=float(np.sqrt(np.mean(dq**2))),
        qvel_l2=float(np.linalg.norm(dv[-1])),
        qvel_max_abs=float(np.abs(dv).max()),
        qvel_rms=float(np.sqrt(np.mean(dv**2))),
        matches_bitwise=bool(np.array_equal(a.qpos, b.qpos) and np.array_equal(a.qvel, b.qvel)),
        matches_1e_8=bool(np.all(np.abs(dq) < 1e-8) and np.all(np.abs(dv) < 1e-8)),
        matches_1e_5=bool(np.all(np.abs(dq) < 1e-5) and np.all(np.abs(dv) < 1e-5)),
        matches_1e_2=bool(np.all(np.abs(dq) < 1e-2) and np.all(np.abs(dv) < 1e-2)),
    )
