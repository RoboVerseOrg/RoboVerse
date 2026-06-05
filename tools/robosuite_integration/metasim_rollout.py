"""Step robosuite's compiled MJCF through MetaSim's model-loading path.

MetaSim's ``MujocoHandler`` composes models with dm_control / PyMJCF
(``mjcf.from_path`` → ``mjcf.Physics.from_mjcf_model``). To test whether that
loader preserves robosuite physics, we load the *identical* exported XML the
same way and step the raw actuators with the *identical* controller-free ctrl
signal, then compare against the robosuite-native rollout.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from .common import CtrlFn, RolloutResult


def metasim_dm_control_rollout(
    xml: str,
    *,
    task_id: str,
    n_steps: int,
    decimation: int,
    ctrl_fn: CtrlFn,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    model_overrides: dict[str, np.ndarray] | None = None,
) -> RolloutResult:
    """Load ``xml`` via dm_control (MetaSim's loader) and roll out controller-free.

    ``model_overrides`` re-applies compiled-model fields robosuite mutates at
    reset (e.g. ``body_pos``/``body_quat`` for welded objects) that ``get_xml()``
    omits — required for exact parity on tasks like Door.
    """
    from dm_control import mjcf

    mjcf_model = mjcf.from_xml_string(xml, assets=None)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)
    m, d = physics.model.ptr, physics.data.ptr

    for field, value in (model_overrides or {}).items():
        getattr(m, field)[:] = value

    mujoco.mj_resetData(m, d)
    if d.qpos.shape == qpos0.shape:
        d.qpos[:] = qpos0
        d.qvel[:] = qvel0
    else:
        raise RuntimeError(f"qpos shape mismatch: dm_control {d.qpos.shape} vs robosuite {qpos0.shape}")
    mujoco.mj_forward(m, d)

    nu = m.nu
    qpos_log = [d.qpos.copy()]
    qvel_log = [d.qvel.copy()]
    ctrl_log: list[np.ndarray] = []
    time_log = [float(d.time)]

    for t in range(n_steps):
        ctrl = np.asarray(ctrl_fn(t, d.qpos.copy(), d.qvel.copy()), dtype=np.float64).reshape(nu)
        d.ctrl[:] = ctrl
        ctrl_log.append(ctrl.copy())
        for _ in range(decimation):
            mujoco.mj_step(m, d)
        qpos_log.append(d.qpos.copy())
        qvel_log.append(d.qvel.copy())
        time_log.append(float(d.time))

    joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"j{i}" for i in range(m.njnt)]
    actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}" for i in range(nu)]
    return RolloutResult(
        backend="metasim_dm_control",
        task_id=task_id,
        dt=float(m.opt.timestep),
        decimation=decimation,
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log),
        time=np.array(time_log),
        joint_names=joint_names,
        actuator_names=actuator_names,
    )
