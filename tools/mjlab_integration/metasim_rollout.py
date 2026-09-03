"""Run a rollout through MetaSim's MuJoCo loader path (dm_control roundtrip).

MetaSim's `MujocoHandler` builds a scene MJCF, runs it through dm_control's
`mjcf.export_with_assets`, and reloads as `mujoco.MjModel`. Any divergence
between raw `MjModel.from_xml_path(orig)` and that round-trip indicates a
loader-side bug — independent of any control or scenario wrapper.

This module exposes two functions:
  * `metasim_loader_rollout`: emulates MetaSim's MJCF round-trip in isolation
    (no scenario/robot wrapping), then steps with raw mujoco.
  * `metasim_full_rollout`: builds a minimal `ScenarioCfg`, launches a real
    `MujocoHandler`, drives it for N control steps, and returns the trace.

Both return `RolloutResult` matching the raw-mujoco harness, so the diff
report can plot them on the same axes.
"""

from __future__ import annotations

import glob
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf as _dm_mjcf

from .raw_rollout import RolloutResult, _set_init_qpos


def _dm_roundtrip(xml_path: Path) -> Path:
    """Load via dm_control, export, return the path to the exported root XML.

    Caller is responsible for keeping the parent tempdir alive (we keep the
    TemporaryDirectory bound to the returned object's `parent`).
    """
    model = _dm_mjcf.from_path(str(xml_path))
    tmpdir = tempfile.mkdtemp(prefix="mjlab_metasim_roundtrip_")
    try:
        _dm_mjcf.export_with_assets(model, out_dir=tmpdir)
    except TypeError:
        _dm_mjcf.export_with_assets(model, tmpdir)
    candidates = sorted(glob.glob(os.path.join(tmpdir, "*.xml")))
    if not candidates:
        fallback = os.path.join(tmpdir, "model.xml")
        with open(fallback, "w") as f:
            f.write(model.to_xml_string())
        candidates = [fallback]
    return Path(candidates[0])


def metasim_loader_rollout(
    xml_path: str | Path,
    *,
    n_steps: int,
    decimation: int,
    ctrl_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    init_joint_pos: dict[str, float] | None = None,
    timestep: float | None = None,
    task_id: str = "metasim_loader",
    seed: int = 0,
) -> RolloutResult:
    """Reload an MJCF through the dm_control export pipeline, then roll out.

    Mirrors `MujocoHandler.launch()`'s "build the scene MJCF, export with
    assets, then `mujoco.MjModel.from_xml_path` the exported XML" sequence.
    """
    exported = _dm_roundtrip(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(exported))
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
        backend="metasim_loader",
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
# Full pipeline rollout: scenario + MujocoHandler.simulate().
# ---------------------------------------------------------------------------


def metasim_full_rollout(
    scenario,
    *,
    n_steps: int,
    decimation: int,
    ctrl_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    init_joint_pos: dict[str, float] | None = None,
    task_id: str = "metasim_full",
    seed: int = 0,
) -> RolloutResult:
    """Launch a MujocoHandler with the given scenario and step it.

    We bypass `_set_dof_targets` (which routes via per-robot joint names) and
    write directly to `physics.data.ctrl` so the diff isolates the *physics*
    step, not the action-routing layer.
    """
    from metasim.sim.mujoco import MujocoHandler  # local import: heavy module

    scenario.decimation = decimation
    scenario.headless = True
    scenario.num_envs = 1

    handler = MujocoHandler(scenario)
    handler.launch()

    physics = handler.physics
    model = physics.model
    data = physics.data

    # Apply init_joint_pos if provided (raw joint-name lookups against the
    # final, dm_control-attached model).
    if init_joint_pos:
        for jname, val in init_joint_pos.items():
            jid = mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                # try suffix match — dm_control attach prefixes with namespace
                full_names = [
                    mujoco.mj_id2name(model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)
                ]
                cand = [i for i, n in enumerate(full_names) if n.endswith("/" + jname) or n == jname]
                if not cand:
                    continue
                jid = cand[0]
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = float(val)
        physics.forward()

    np.random.seed(seed)

    qpos_log = [np.array(data.qpos, copy=True)]
    qvel_log = [np.array(data.qvel, copy=True)]
    ctrl_log: list[np.ndarray] = []
    time_log = [float(data.time)]

    nu = int(model.nu)

    for t in range(n_steps):
        ctrl = ctrl_fn(t, np.array(data.qpos), np.array(data.qvel))
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(nu)
        data.ctrl[:] = ctrl
        ctrl_log.append(ctrl.copy())
        handler._simulate()
        qpos_log.append(np.array(data.qpos, copy=True))
        qvel_log.append(np.array(data.qvel, copy=True))
        time_log.append(float(data.time))

    joint_names = [mujoco.mj_id2name(model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt_{i}" for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model.ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}" for i in range(nu)]

    try:
        handler.close()
    except Exception:
        pass

    return RolloutResult(
        backend="metasim_full",
        task_id=task_id,
        dt=float(model.opt.timestep),
        decimation=decimation,
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log) if ctrl_log else np.zeros((0, nu)),
        time=np.array(time_log),
        joint_names=joint_names,
        actuator_names=actuator_names,
        jnt_qposadr=[int(x) for x in model.jnt_qposadr],
        jnt_dofadr=[int(x) for x in model.jnt_dofadr],
        jnt_type=[int(x) for x in model.jnt_type],
    )
