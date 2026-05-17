"""Drill into per-geom properties + exported MJCF for cartpole.

The structural diff showed MetaSim adds an unnamed geom 0 (likely a ground
plane). We need to see whether it touches the cart/pole and what its
contype/conaffinity/friction are, since the sim diverges drastically (qpos
max|Δ|≈6.5 rad) despite identical nq/nu/timestep/integrator/gravity.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from . import inventory as inv
from .full_runner import SCENARIO_BUILDERS


def _per_geom(model: mujoco.MjModel) -> list[dict]:
    out = []
    for i in range(model.ngeom):
        out.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "<unnamed>",
            "type": int(model.geom_type[i]),
            "pos": [float(x) for x in model.geom_pos[i]],
            "size": [float(x) for x in model.geom_size[i]],
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
            "bodyid": int(model.geom_bodyid[i]),
            "friction": [float(x) for x in model.geom_friction[i]],
        })
    return out


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    task = inv.task_by_metasim_name("mjlab.cartpole_balance")

    raw = mujoco.MjModel.from_xml_path(str(task.asset.xml_path))
    raw_g = _per_geom(raw)
    raw_body_names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, i) or f"b{i}" for i in range(raw.nbody)]

    builder = SCENARIO_BUILDERS["mjlab.cartpole_balance"]
    scenario = builder(task)
    scenario.sim_params.dt = float(raw.opt.timestep)

    from metasim.sim.mujoco import MujocoHandler

    handler = MujocoHandler(scenario)
    handler.launch()
    meta = handler.physics.model.ptr
    meta_g = _per_geom(meta)
    meta_body_names = [mujoco.mj_id2name(meta, mujoco.mjtObj.mjOBJ_BODY, i) or f"b{i}" for i in range(meta.nbody)]

    print("=== RAW geoms ===")
    for g in raw_g:
        bn = raw_body_names[g["bodyid"]]
        print(
            f"  [{g['id']}] name={g['name']!r:<30} type={g['type']} pos={g['pos']} size={g['size']} contype={g['contype']} conaffinity={g['conaffinity']} body={bn!r}"
        )

    print("\n=== META geoms ===")
    for g in meta_g:
        bn = meta_body_names[g["bodyid"]]
        print(
            f"  [{g['id']}] name={g['name']!r:<30} type={g['type']} pos={g['pos']} size={g['size']} contype={g['contype']} conaffinity={g['conaffinity']} body={bn!r}"
        )

    # Run with raw mj_step on both models (bypass dm_control MjData wrapper)
    print("\n=== one decimation block, step by step (ctrl=0.3) ===")
    rd = mujoco.MjData(raw)
    mujoco.mj_resetData(raw, rd)
    mujoco.mj_forward(raw, rd)

    md = mujoco.MjData(meta)
    mujoco.mj_resetData(meta, md)
    mujoco.mj_forward(meta, md)

    print(f"  t=0  raw qpos={rd.qpos.copy()}  qvel={rd.qvel.copy()}")
    print(f"  t=0  meta qpos={md.qpos.copy()}  qvel={md.qvel.copy()}")

    for k in range(20):
        rd.ctrl[:] = 0.3
        md.ctrl[:] = 0.3
        mujoco.mj_step(raw, rd)
        mujoco.mj_step(meta, md)
        diff_q = np.abs(rd.qpos - md.qpos).max()
        diff_v = np.abs(rd.qvel - md.qvel).max()
        marker = "" if diff_q < 1e-9 else "  <-- DIFF"
        print(f"  k={k:>2}  raw qpos={rd.qpos}  meta qpos={md.qpos}  |Δq|={diff_q:.3e}  |Δv|={diff_v:.3e}{marker}")

    # Dump per-joint damping/armature/frictionloss/solref/solimp comparison
    print("\n=== joint dynamics params ===")
    for jid in range(raw.njnt):
        rname = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, jid)
        mname = mujoco.mj_id2name(meta, mujoco.mjtObj.mjOBJ_JOINT, jid)
        print(f"  joint[{jid}] raw={rname!r} meta={mname!r}")
        rdof = raw.jnt_dofadr[jid]
        mdof = meta.jnt_dofadr[jid]
        print(f"     damping       raw={raw.dof_damping[rdof]:.6e}  meta={meta.dof_damping[mdof]:.6e}")
        print(f"     armature      raw={raw.dof_armature[rdof]:.6e}  meta={meta.dof_armature[mdof]:.6e}")
        print(f"     frictionloss  raw={raw.dof_frictionloss[rdof]:.6e}  meta={meta.dof_frictionloss[mdof]:.6e}")
        print(f"     limited       raw={raw.jnt_limited[jid]}  meta={meta.jnt_limited[jid]}")
        print(f"     range         raw={raw.jnt_range[jid]}  meta={meta.jnt_range[jid]}")
        print(f"     solref_limit  raw={raw.jnt_solref[jid]}  meta={meta.jnt_solref[jid]}")
        print(f"     solimp_limit  raw={raw.jnt_solimp[jid]}  meta={meta.jnt_solimp[jid]}")
        print(f"     margin        raw={raw.jnt_margin[jid]:.6e}  meta={meta.jnt_margin[jid]:.6e}")

    # Body inertials / mass
    print("\n=== body inertial params ===")
    for bid in range(min(raw.nbody, meta.nbody)):
        rname = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, bid)
        mname = mujoco.mj_id2name(meta, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  body[{bid}] raw={rname!r} meta={mname!r}")
        print(f"     mass  raw={raw.body_mass[bid]:.6e}  meta={meta.body_mass[bid]:.6e}")
        print(f"     inertia raw={raw.body_inertia[bid]}  meta={meta.body_inertia[bid]}")
        print(f"     ipos  raw={raw.body_ipos[bid]}  meta={meta.body_ipos[bid]}")

    handler.close()


if __name__ == "__main__":
    main()
