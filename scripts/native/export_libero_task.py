"""Export a LIBERO task to a self-contained MetaSim-native bundle (uses libero ONCE).

Produces, under ``roboverse_pack/tasks/libero_native/assets/<name>/``:
  * ``model.mjb``       — the compiled MuJoCo model (binary; embeds meshes +
                          textures, so it loads with NO asset files and NO libero)
  * ``init_states.npy`` — demo initial flat states
  * ``demo_actions.npy`` — a demo's action sequence (to drive the native rollout)
  * ``goal.json``       — the BDDL goal as resolved predicate dicts for the
                          ported checker (model body/site names already resolved)

It also **verifies in-line** that the ported checker (no libero) returns the same
per-step success as the live ``env._check_success()`` along a demo rollout — so we
know the bundle is faithful before anything runs natively.

Run (liberoplus env):
    LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \\
    python -m scripts.native.export_libero_task --suite libero_object \\
        --base pick_up_the_alphabet_soup_and_place_it_in_the_basket --name alphabet_soup
"""

from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np

from roboverse_pack.tasks.libero_native import checker as nc
from roboverse_pack.tasks.libero_plus import _passthrough as pt

ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "roboverse_pack",
    "tasks",
    "libero_native",
    "assets",
)
DEMO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "libero_datasets"
)


def _resolve_goal(e):
    """Turn parsed BDDL goal_state into predicate dicts with model names."""
    m = e.sim.model._model
    goal = []
    for state in e.parsed_problem["goal_state"]:
        fn = state[0].lower()
        if fn == "in":
            obj, region = state[1], state[2]  # arg2 is a region site
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, e.obj_body_id[obj])
            goal.append({"fn": "in", "obj": body_name, "region": region})
        elif fn == "on":
            a, b = state[1], state[2]
            goal.append({
                "fn": "on",
                "obj": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, e.obj_body_id[a]),
                "obj2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, e.obj_body_id[b]),
            })
        else:
            raise NotImplementedError(f"goal predicate {fn!r} not yet supported for export")
    return goal


def run(suite, base, name, n_frames):
    names = pt.list_liberoplus_tasks(suite)
    tid = next(i for i, n in enumerate(names) if n.startswith(base))
    env = pt.make_liberoplus_env(suite, tid, seed=0)
    e = env.env

    goal = _resolve_goal(e)
    print(f"goal: {goal}")

    # OSC config + actuator layout (so the native task needs no libero/robosuite)
    r, c = e.robots[0], e.robots[0].controller
    nu = e.sim.model.nu
    arm_act = list(r._ref_joint_actuator_indexes)
    cfg = {
        "eef_site": c.eef_name,
        "arm_qvel_index": list(map(int, c.qvel_index)),
        "arm_act_index": list(map(int, arm_act)),
        "grip_act_index": [i for i in range(nu) if i not in arm_act],
        "initial_joint": list(map(float, np.array(c.initial_joint))),
        "torque_limits": [list(map(float, r.torque_limits[0])), list(map(float, r.torque_limits[1]))],
        "substeps": int(e.control_timestep / e.model_timestep),
    }

    with __import__("h5py").File(os.path.join(DEMO_ROOT, suite, f"{base}_demo.hdf5"), "r") as h:
        actions = np.asarray(h["data"]["demo_0"]["actions"])[:n_frames]
        init_state = np.asarray(h["data"]["demo_0"]["states"])[0]

    # in-line verify: ported checker vs native env._check_success along the rollout;
    # also capture the per-step gripper ctrl so the native replay matches exactly.
    m, d = e.sim.model._model, e.sim.data._data
    env.set_init_state(init_state)
    mism, n_succ_native, n_succ_ported = 0, 0, 0
    grip_ctrl = []
    for a in actions:
        env.step(a.tolist())
        grip_ctrl.append(np.array(e.sim.data.ctrl[cfg["grip_act_index"]]))
        s_native = bool(e._check_success())
        s_ported = nc.check_success(m, d, goal)
        mism += int(s_native != s_ported)
        n_succ_native += int(s_native)
        n_succ_ported += int(s_ported)
    print(
        f"checker parity: {mism} per-step mismatches over {len(actions)} steps "
        f"(native success steps={n_succ_native}, ported={n_succ_ported})"
    )

    # export the self-contained bundle
    out = os.path.join(ASSETS, name)
    os.makedirs(out, exist_ok=True)
    mujoco.mj_saveModel(e.sim.model._model, os.path.join(out, "model.mjb"), None)
    np.save(os.path.join(out, "init_states.npy"), init_state)
    np.save(os.path.join(out, "demo_actions.npy"), actions)
    np.save(os.path.join(out, "grip_ctrl.npy"), np.array(grip_ctrl))
    json.dump(
        {"goal": goal, "suite": suite, "base": base, "task": names[tid], "osc": cfg},
        open(os.path.join(out, "goal.json"), "w"),
        indent=2,
    )
    env.close()
    sz = os.path.getsize(os.path.join(out, "model.mjb")) // 1024
    print(f"exported -> {out} (model.mjb {sz} KB, init_states, demo_actions, goal.json)")
    return 0 if mism == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--base", default="pick_up_the_alphabet_soup_and_place_it_in_the_basket")
    ap.add_argument("--name", default="alphabet_soup")
    ap.add_argument("--frames", type=int, default=140)
    args = ap.parse_args()
    return run(args.suite, args.base, args.name, args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
