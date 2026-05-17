"""Side-by-side structural diff of raw cartpole MJCF vs MetaSim's wrapped model.

We dump model.body_names / joint_names / actuator_names / qpos addresses,
plus key scalar physics options (timestep, gravity, friction defaults,
solver iterations) and any joints introduced by the MetaSim wrapper.

This narrows the cartpole divergence root-cause hypothesis (extra freejoint,
ground geom, actuator gear, ...) before we patch anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from . import inventory as inv
from .full_runner import SCENARIO_BUILDERS


def _names(model: mujoco.MjModel, obj_type, n: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj_type, i) or f"<unnamed_{i}>" for i in range(n)]


def _dump_model(label: str, model: mujoco.MjModel) -> dict:
    info = {
        "label": label,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "timestep": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "iterations": int(model.opt.iterations),
        "solver": int(model.opt.solver),
        "integrator": int(model.opt.integrator),
        "body_names": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joint_names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "joint_types": [int(t) for t in model.jnt_type],
        "jnt_qposadr": [int(a) for a in model.jnt_qposadr],
        "jnt_dofadr": [int(a) for a in model.jnt_dofadr],
        "actuator_names": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "actuator_gear": [[float(x) for x in g] for g in model.actuator_gear],
        "actuator_trnid": [[int(x) for x in a] for a in model.actuator_trnid],
        "geom_names": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
    }
    return info


def diagnose_cartpole():
    os.environ.setdefault("MUJOCO_GL", "egl")
    task = inv.task_by_metasim_name("mjlab.cartpole_balance")

    raw = mujoco.MjModel.from_xml_path(str(task.asset.xml_path))
    raw_info = _dump_model("raw", raw)

    builder = SCENARIO_BUILDERS["mjlab.cartpole_balance"]
    scenario = builder(task)
    scenario.sim_params.dt = float(raw.opt.timestep)

    from metasim.sim.mujoco import MujocoHandler

    handler = MujocoHandler(scenario)
    handler.launch()
    meta = handler.physics.model.ptr
    meta_info = _dump_model("metasim_full", meta)

    print("=" * 80)
    print(f"{'field':<22} {'raw':<32} metasim_full")
    print("-" * 80)
    keys_scalar = [
        "nq",
        "nv",
        "nu",
        "njnt",
        "nbody",
        "ngeom",
        "timestep",
        "gravity",
        "iterations",
        "solver",
        "integrator",
    ]
    for k in keys_scalar:
        a, b = raw_info[k], meta_info[k]
        marker = "" if a == b else "  <-- DIFF"
        print(f"{k:<22} {a!s:<32} {b!s}{marker}")

    print("\n=== bodies ===")
    for i, n in enumerate(raw_info["body_names"]):
        print(f"  raw[{i:>2}]   {n}")
    for i, n in enumerate(meta_info["body_names"]):
        print(f"  meta[{i:>2}]  {n}")

    print("\n=== joints (id, type, qposadr, dofadr, name) ===")
    for i in range(raw_info["njnt"]):
        print(
            f"  raw [{i}]  type={raw_info['joint_types'][i]} "
            f"qposadr={raw_info['jnt_qposadr'][i]} dofadr={raw_info['jnt_dofadr'][i]}  "
            f"name={raw_info['joint_names'][i]!r}"
        )
    for i in range(meta_info["njnt"]):
        print(
            f"  meta[{i}]  type={meta_info['joint_types'][i]} "
            f"qposadr={meta_info['jnt_qposadr'][i]} dofadr={meta_info['jnt_dofadr'][i]}  "
            f"name={meta_info['joint_names'][i]!r}"
        )

    print("\n=== actuators (id, gear, trnid, name) ===")
    for i in range(raw_info["nu"]):
        print(
            f"  raw [{i}]  gear={raw_info['actuator_gear'][i]}  trnid={raw_info['actuator_trnid'][i]}  "
            f"name={raw_info['actuator_names'][i]!r}"
        )
    for i in range(meta_info["nu"]):
        print(
            f"  meta[{i}]  gear={meta_info['actuator_gear'][i]}  trnid={meta_info['actuator_trnid'][i]}  "
            f"name={meta_info['actuator_names'][i]!r}"
        )

    print("\n=== geoms (id, name) ===")
    for i, n in enumerate(raw_info["geom_names"]):
        print(f"  raw [{i:>2}]  {n}")
    for i, n in enumerate(meta_info["geom_names"]):
        print(f"  meta[{i:>2}]  {n}")

    # Dump exported xml for a separate diff if needed.
    tmpdir = Path("/tmp/mjlab_diag_cartpole")
    tmpdir.mkdir(exist_ok=True)
    try:
        with open(tmpdir / "metasim_export.xml", "w") as f:
            mujoco.mj_saveLastXML(str(tmpdir / "metasim_export.xml"), meta)
    except Exception:
        # mj_saveLastXML only works for last-loaded; fall back to to_xml_string via the dm_control model
        try:
            from dm_control import mjcf

            with open(tmpdir / "metasim_dm.xml", "w") as f:
                f.write(mjcf.from_path(str(task.asset.xml_path)).to_xml_string())
        except Exception as e:
            print(f"(could not dump xml: {e})")
    print(f"\n(wrote any available xml dumps to {tmpdir})")

    try:
        handler.close()
    except Exception:
        pass

    return raw_info, meta_info


if __name__ == "__main__":
    diagnose_cartpole()
