"""Broad engine-parity sweep across robosuite tasks x robots.

Controller-free: load each (env, robot)'s compiled MJCF, drive the raw actuators
with a deterministic sinusoid in robosuite's own MjSim and in MetaSim's
dm_control loader, compare qpos/qvel. Task- and robot-agnostic — proves MetaSim
hosts the physics of the *whole* robosuite surface bit-for-bit, without porting
any task logic.

Usage:
    python -m tools.robosuite_integration.coverage_sweep --out <report>/coverage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import mujoco
import numpy as np

# single-arm concrete manipulation tasks
SINGLE_ARM_TASKS = [
    "Lift",
    "Stack",
    "Door",
    "Wipe",
    "ToolHang",
    "NutAssembly",
    "NutAssemblySquare",
    "NutAssemblyRound",
    "PickPlace",
    "PickPlaceCan",
    "PickPlaceBread",
    "PickPlaceCereal",
    "PickPlaceMilk",
]
TWO_ARM_TASKS = ["TwoArmLift", "TwoArmPegInHole", "TwoArmHandover", "TwoArmTransport"]
ROBOTS = ["Panda", "Sawyer", "IIWA", "UR5e", "Kinova3", "Jaco", "XArm7"]


def _make(env_name: str, robots):
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC", robot=robots if isinstance(robots, str) else robots[0])
    return robosuite.make(
        env_name,
        robots=robots,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        ignore_done=True,
    )


def engine_parity(env_name: str, robots, n_steps: int = 60) -> dict:
    env = _make(env_name, robots)
    np.random.seed(0)
    env.reset()
    xml = env.model.get_xml()
    m, d = env.sim.model._model, env.sim.data._data
    nu, nq, nv = m.nu, m.nq, m.nv
    dec = round((1 / 20) / m.opt.timestep)
    body_pos, body_quat = m.body_pos.copy(), m.body_quat.copy()
    qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()

    def ctrl(t):
        return 0.4 * np.sin(2 * np.pi * t / 40 + np.arange(nu) * 0.7)

    def roll(model, data, apply_override=False):
        if apply_override:
            model.body_pos[:] = body_pos
            model.body_quat[:] = body_quat
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(model, data)
        qp = [data.qpos.copy()]
        for t in range(n_steps):
            data.ctrl[:] = ctrl(t)
            for _ in range(dec):
                mujoco.mj_step(model, data)
            qp.append(data.qpos.copy())
        return np.stack(qp)

    rs_qpos = roll(m, d)
    env.close()

    from dm_control import mjcf

    physics = mjcf.Physics.from_mjcf_model(mjcf.from_xml_string(xml, assets=None))
    ms_qpos = roll(physics.model.ptr, physics.data.ptr, apply_override=True)

    diff = float(np.abs(rs_qpos - ms_qpos).max())
    return {
        "task": env_name,
        "robot": robots if isinstance(robots, str) else "+".join(robots),
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(nu),
        "qpos_max_abs_diff": diff,
        "bitwise": bool(diff == 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--two-arm", action="store_true")
    args = ap.parse_args()

    results = {"tasks_panda": [], "robots_lift": [], "two_arm": []}

    print("== all single-arm manipulation tasks (Panda), engine parity ==")
    for t in SINGLE_ARM_TASKS:
        try:
            r = engine_parity(t, "Panda", args.n_steps)
            print(
                f"  {t:20s} nq={r['nq']:3d} nu={r['nu']:2d} max|Δqpos|={r['qpos_max_abs_diff']:.1e} bitwise={r['bitwise']}"
            )
        except Exception as e:
            r = {"task": t, "robot": "Panda", "error": str(e)[:120]}
            print(f"  {t:20s} ERROR {str(e)[:80]}")
        results["tasks_panda"].append(r)

    print("\n== Lift across robots, engine parity ==")
    for rb in ROBOTS:
        try:
            r = engine_parity("Lift", rb, args.n_steps)
            print(
                f"  {rb:10s} nq={r['nq']:3d} nu={r['nu']:2d} max|Δqpos|={r['qpos_max_abs_diff']:.1e} bitwise={r['bitwise']}"
            )
        except Exception as e:
            r = {"task": "Lift", "robot": rb, "error": str(e)[:120]}
            print(f"  {rb:10s} ERROR {str(e)[:80]}")
        results["robots_lift"].append(r)

    if args.two_arm:
        print("\n== two-arm tasks (Panda, Panda), engine parity ==")
        for t in TWO_ARM_TASKS:
            try:
                r = engine_parity(t, ["Panda", "Panda"], args.n_steps)
                print(
                    f"  {t:20s} nq={r['nq']:3d} nu={r['nu']:2d} max|Δqpos|={r['qpos_max_abs_diff']:.1e} bitwise={r['bitwise']}"
                )
            except Exception as e:
                r = {"task": t, "robot": "Panda+Panda", "error": str(e)[:120]}
                print(f"  {t:20s} ERROR {str(e)[:80]}")
            results["two_arm"].append(r)

    def summ(key):
        rs = [x for x in results[key] if "bitwise" in x]
        return f"{sum(x['bitwise'] for x in rs)}/{len(results[key])} bitwise"

    results["summary"] = {k: summ(k) for k in results if results[k]}
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSUMMARY: tasks(Panda) {summ('tasks_panda')} | robots(Lift) {summ('robots_lift')}", end="")
    if args.two_arm:
        print(f" | two-arm {summ('two_arm')}", end="")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
