"""Quantify divergence between two engine rollouts of the same model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .common import RolloutResult


@dataclass
class DiffStats:
    """Numerical agreement between two (aligned) rollouts."""

    task_id: str
    backend_a: str
    backend_b: str
    T: int
    nq: int
    nv: int
    qpos_max_abs: float
    qpos_rms: float
    qpos_final_max_abs: float
    qvel_max_abs: float
    qvel_rms: float
    bitwise_equal: bool
    matches_1e_12: bool
    matches_1e_9: bool
    matches_1e_6: bool
    matches_1e_3: bool

    def to_dict(self) -> dict:
        return asdict(self)


def diff_rollouts(a: RolloutResult, b: RolloutResult) -> DiffStats:
    """Compare qpos/qvel of two rollouts assumed to share model layout."""
    if a.qpos.shape != b.qpos.shape:
        raise ValueError(f"qpos shape mismatch: {a.qpos.shape} vs {b.qpos.shape}")
    dq = np.abs(a.qpos - b.qpos)
    dv = np.abs(a.qvel - b.qvel)
    qpos_max = float(dq.max())
    qvel_max = float(dv.max())
    return DiffStats(
        task_id=a.task_id,
        backend_a=a.backend,
        backend_b=b.backend,
        T=int(a.qpos.shape[0] - 1),
        nq=int(a.qpos.shape[1]),
        nv=int(a.qvel.shape[1]),
        qpos_max_abs=qpos_max,
        qpos_rms=float(np.sqrt(np.mean(dq**2))),
        qpos_final_max_abs=float(dq[-1].max()),
        qvel_max_abs=qvel_max,
        qvel_rms=float(np.sqrt(np.mean(dv**2))),
        bitwise_equal=bool((a.qpos == b.qpos).all() and (a.qvel == b.qvel).all()),
        matches_1e_12=bool(qpos_max <= 1e-12 and qvel_max <= 1e-12),
        matches_1e_9=bool(qpos_max <= 1e-9 and qvel_max <= 1e-9),
        matches_1e_6=bool(qpos_max <= 1e-6 and qvel_max <= 1e-6),
        matches_1e_3=bool(qpos_max <= 1e-3 and qvel_max <= 1e-3),
    )
