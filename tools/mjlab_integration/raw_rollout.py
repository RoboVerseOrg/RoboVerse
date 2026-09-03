"""Roll out an mjlab MJCF in pure MuJoCo (CPU).

This is the *reference* trajectory the MetaSim port must match. Pure MuJoCo
loads the same XML mjlab loads; the only differences from mjlab itself are
that we don't go through the mujoco-warp GPU kernels and we don't apply
mjlab's manager-based observation/event/reward wrappers. Physics state
evolution is the same.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class RolloutResult:
    """Time-series captured from one rollout."""

    backend: str  # "raw_mujoco" or "metasim_mujoco"
    task_id: str
    dt: float  # base physics timestep
    decimation: int  # ctrl-application substeps per control step
    qpos: np.ndarray  # (T+1, nq) — includes initial frame
    qvel: np.ndarray  # (T+1, nv)
    ctrl: np.ndarray  # (T, nu) — applied each control step
    time: np.ndarray  # (T+1,)
    joint_names: list[str]
    actuator_names: list[str]
    jnt_qposadr: list[int] | None = None  # MjModel.jnt_qposadr per joint
    jnt_dofadr: list[int] | None = None  # MjModel.jnt_dofadr per joint
    jnt_type: list[int] | None = None  # mjtJoint per joint (free/ball/slide/hinge)

    def summary(self) -> dict[str, float]:
        return {
            "T": int(self.qpos.shape[0] - 1),
            "nq": int(self.qpos.shape[1]),
            "nv": int(self.qvel.shape[1]),
            "nu": int(self.ctrl.shape[1]),
            "max_abs_qpos": float(np.abs(self.qpos).max()),
            "max_abs_qvel": float(np.abs(self.qvel).max()),
        }


def _set_init_qpos(model: mujoco.MjModel, data: mujoco.MjData, init_pos: dict[str, float]) -> None:
    for jname, val in init_pos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = float(val)


def raw_mujoco_rollout(
    xml_path: str | Path,
    *,
    n_steps: int,
    decimation: int,
    ctrl_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    init_joint_pos: dict[str, float] | None = None,
    timestep: float | None = None,
    task_id: str = "raw",
    seed: int = 0,
) -> RolloutResult:
    """Step a raw mujoco model `n_steps` control updates by `decimation` substeps.

    Args:
        xml_path: path to MJCF.
        n_steps: number of control steps (each writes a fresh ctrl, then we
            `mj_step` `decimation` times before recording).
        decimation: physics substeps per control step.
        ctrl_fn: callable (t, qpos, qvel) → ctrl of shape (nu,). Determines the
            policy. Use a constant (e.g. zero) for purely-passive determinism
            checks, or a sinusoid for active comparisons.
        init_joint_pos: optional joint-name → qpos overrides applied after
            mj_resetData.
        timestep: optional override for the model timestep.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if timestep is not None:
        model.opt.timestep = float(timestep)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if init_joint_pos:
        _set_init_qpos(model, data, init_joint_pos)
    mujoco.mj_forward(model, data)

    np.random.seed(seed)

    qpos_log = [data.qpos.copy()]
    qvel_log = [data.qvel.copy()]
    ctrl_log: list[np.ndarray] = []
    time_log = [float(data.time)]

    for t in range(n_steps):
        ctrl = ctrl_fn(t, data.qpos.copy(), data.qvel.copy())
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(model.nu)
        data.ctrl[:] = ctrl
        ctrl_log.append(ctrl.copy())
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        time_log.append(float(data.time))

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt_{i}" for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}" for i in range(model.nu)]

    return RolloutResult(
        backend="raw_mujoco",
        task_id=task_id,
        dt=float(model.opt.timestep),
        decimation=decimation,
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log) if ctrl_log else np.zeros((0, model.nu)),
        time=np.array(time_log),
        joint_names=joint_names,
        actuator_names=actuator_names,
        jnt_qposadr=[int(x) for x in model.jnt_qposadr],
        jnt_dofadr=[int(x) for x in model.jnt_dofadr],
        jnt_type=[int(x) for x in model.jnt_type],
    )


# ---------------------------------------------------------------------------
# Standard ctrl policies (deterministic).
# ---------------------------------------------------------------------------


def zero_ctrl(nu: int) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
    z = np.zeros(nu)

    def _fn(t, q, v):
        return z

    return _fn


def sinusoid_ctrl(nu: int, amplitude: float = 0.3, freq_hz: float = 0.5, dt: float = 0.01) -> Callable:
    """One-period sinusoid; deterministic across runs."""

    def _fn(t, q, v):
        phase = 2.0 * math.pi * freq_hz * (t * dt)
        return amplitude * np.sin(phase + np.linspace(0.0, math.pi / 2, nu))

    return _fn


def stationary_swingup_ctrl(nu: int) -> Callable:
    """Just hold zero. Useful for swing-up: pole should fall under gravity."""
    return zero_ctrl(nu)
