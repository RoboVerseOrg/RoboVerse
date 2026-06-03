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


# open/close/turnon/turnoff -> (object-state method, any/all reduction)
_JOINT_PRED = {
    "open": ("is_open", "any"),
    "close": ("is_close", "all"),
    "turnon": ("turn_on", "any"),
    "turnoff": ("turn_off", "all"),
}


def _probe_threshold(method, ranges):
    """Recover ``(op, threshold)`` from an articulated object's own boolean method.

    Each ``is_open``/``is_close``/``turn_on``/``turn_off`` is a single comparison
    ``qpos {<,<=,>,>=} T`` (see ``articulated_objects.py``). We call the live method
    (so the result is exactly faithful) at extremes to get the direction, then find
    ``T`` among the object's range boundaries and the strictness by probing ``T``.
    """
    lo, hi = bool(method(-1e9)), bool(method(1e9))
    cand = sorted({float(v) for lst in ranges.values() for v in (lst or [])})
    thr = next((c for c in cand if bool(method(c - 1e-7)) != bool(method(c + 1e-7))), None)
    if thr is None:
        raise RuntimeError(f"could not locate threshold among {cand}")
    if lo and not hi:  # True for small qpos -> "<" form
        op = "le" if method(thr) else "lt"
    elif hi and not lo:  # True for large qpos -> ">" form
        op = "ge" if method(thr) else "gt"
    else:
        raise RuntimeError(f"method is not a monotone step (lo={lo}, hi={hi})")
    return op, float(thr)


def _resolve_joint_predicate(e, fn, name):
    """Resolve open/close/turnon/turnoff to qpos addresses + op/threshold + reduce."""
    method_name, reduce = _JOINT_PRED[fn]
    os_obj = e.object_states_dict[name]
    if getattr(os_obj, "parent_name", None) is not None:  # SiteObjectState
        joints = e.object_sites_dict[name].joints
        owner = e.get_object(os_obj.parent_name)
    else:  # ObjectState
        joints = e.get_object(name).joints
        owner = e.get_object(name)
    op, thr = _probe_threshold(getattr(owner, method_name), owner.object_properties["articulation"])
    addrs = [int(e.sim.model.get_joint_qpos_addr(j)) for j in joints]
    return {"fn": "joint", "reduce": reduce, "op": op, "thr": thr, "qpos_addr": addrs}


def _contact_geom_ids(e, obj_name):
    """The robosuite ``contact_geoms`` of an object, resolved to model geom ids.

    Mirrors ``MujocoEnv.check_contact`` (which uses ``model.contact_geoms``), so the
    native checker tests contact over the exact same geom set LIBERO does.
    """
    m = e.sim.model._model
    ids = []
    for g in e.get_object(obj_name).contact_geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if gid >= 0:
            ids.append(int(gid))
    return ids


def _resolve_goal(e):
    """Turn parsed BDDL goal_state into self-contained predicate dicts.

    Each object is resolved to its model body name; each contact test is resolved to
    explicit geom-id sets; articulated predicates to qpos addresses + op/threshold —
    so the native checker needs only the names + the MuJoCo state, no libero.
    """
    m = e.sim.model._model

    def body(nm):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, e.obj_body_id[nm])

    def is_site(nm):
        return getattr(e.object_states_dict.get(nm), "object_state_type", None) == "site"

    goal = []
    for state in e.parsed_problem["goal_state"]:
        fn = state[0].lower()
        if fn == "in":  # In(obj, region-site) -> in_box (site contact is always True)
            obj, region = state[1], state[2]
            if not is_site(region):
                raise NotImplementedError(f"`in` with non-site arg2 {region!r} not supported")
            goal.append({"fn": "in", "obj": body(obj), "region": region})
        elif fn == "on":
            a, b = state[1], state[2]  # On(arg1=placed, arg2=surface)
            if is_site(b):  # SiteObjectState.check_ontop -> under() + parent contact
                parent = e.object_states_dict[b].parent_name
                # parent may be a region with no backing object (e.g. a table region):
                # LIBERO's check_ontop then skips the contact term and uses under() only.
                parent_obj = e.get_object(parent) if parent is not None else None
                goal.append({
                    "fn": "on_site",
                    "obj": body(a),
                    "site": b,
                    "obj_geoms": _contact_geom_ids(e, a),
                    "parent_geoms": _contact_geom_ids(e, parent) if parent_obj is not None else None,
                })
            else:  # ObjectState.check_ontop -> z + contact + xy
                goal.append({
                    "fn": "on",
                    "obj": body(a),
                    "obj2": body(b),
                    "obj_geoms": _contact_geom_ids(e, a),
                    "obj2_geoms": _contact_geom_ids(e, b),
                })
        elif fn in _JOINT_PRED:
            goal.append(_resolve_joint_predicate(e, fn, state[1]))
        else:
            raise NotImplementedError(f"goal predicate {fn!r} not yet supported for export")
    return goal


def build_osc_cfg(e):
    """Extract the OSC_POSE config + actuator layout from a live LIBERO env.

    Captures everything the native task needs to reconstruct the controller with
    no libero/robosuite import: EE site, arm qvel/actuator indices, gripper
    actuators, initial joint pose, torque limits, and the control:model substep
    ratio (LIBERO's 25 physics steps per policy step).
    """
    r, c = e.robots[0], e.robots[0].controller
    nu = e.sim.model.nu
    arm_act = list(r._ref_joint_actuator_indexes)
    return {
        "eef_site": c.eef_name,
        "arm_qvel_index": list(map(int, c.qvel_index)),
        "arm_act_index": list(map(int, arm_act)),
        "grip_act_index": [i for i in range(nu) if i not in arm_act],
        "initial_joint": list(map(float, np.array(c.initial_joint))),
        "torque_limits": [list(map(float, r.torque_limits[0])), list(map(float, r.torque_limits[1]))],
        "substeps": int(e.control_timestep / e.model_timestep),
    }


def run(suite, base, name, n_frames):
    names = pt.list_liberoplus_tasks(suite)
    tid = next(i for i, n in enumerate(names) if n.startswith(base))
    env = pt.make_liberoplus_env(suite, tid, seed=0)
    e = env.env

    goal = _resolve_goal(e)
    print(f"goal: {goal}")

    # OSC config + actuator layout (so the native task needs no libero/robosuite)
    cfg = build_osc_cfg(e)

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
