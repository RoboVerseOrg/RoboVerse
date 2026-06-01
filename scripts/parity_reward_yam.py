"""Per-reward-term parity: native mjlab YAM lift_cube vs RoboVerse v2.

Builds BOTH the native mjlab ``yam_lift_cube`` env and the RoboVerse v2
``mjlab.lift_cube_yam_v2`` env, injects K matched states (arm+finger joint
state + cube freejoint pose + goal/target command + last_action) into each,
and compares every reward term value at float-eps.

Method:
  - Enumerate native reward terms (name, weight, params, func) from the live
    mjlab ``reward_manager`` and RV terms from the configclass.
  - For each matched state, write identical arm joints, finger joints, cube
    freejoint pose into both physics buffers; set the SAME target_pos on both
    command managers; copy native action/prev_action into RV for action_rate.
  - Evaluate each shared term on both sides, report per-term max|Δ|.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 python scripts/parity_reward_yam.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_obs_all import (  # noqa: E402
    DEVICE,
    _yam_builder,
    build_mjlab,
    build_rv,
)

K = 32
SEED = 3
TOL = 1e-5
N_ARM = 6
N_FINGER = 2
N_JOINT = N_ARM + N_FINGER  # 8


# ---------------------------------------------------------------------------
# term tables
# ---------------------------------------------------------------------------
def native_term_table(mj):
    rm = mj.reward_manager
    out = {}
    for name, cfg in zip(rm.active_terms, rm._term_cfgs):
        out[name] = (cfg.weight, cfg.params, cfg.func)
    return out


def rv_term_table(rv):
    out = {}
    for name in rv._term_names(rv.cfg.rewards):
        term = getattr(rv.cfg.rewards, name)
        out[name] = (term.weight, term.params or {}, term.func)
    return out


def compare_term_sets(native_tbl, rv_tbl):
    print("\n===== YAM: term/weight comparison (native vs RV) =====")
    all_names = list(dict.fromkeys(list(native_tbl) + list(rv_tbl)))
    print(f"{'term':26s} {'native_w':>12s} {'rv_w':>12s}  flag")
    mismatches = []
    for t in all_names:
        nw = native_tbl[t][0] if t in native_tbl else None
        rw = rv_tbl[t][0] if t in rv_tbl else None
        flag = ""
        if nw is None:
            flag = "RV-ONLY"
            mismatches.append((t, "rv-only", nw, rw))
        elif rw is None:
            flag = "NATIVE-ONLY (missing in RV)"
            mismatches.append((t, "missing", nw, rw))
        elif abs(float(nw) - float(rw)) > 1e-12:
            flag = "WEIGHT MISMATCH"
            mismatches.append((t, "weight", nw, rw))
        print(f"{t:26s} {str(nw):>12s} {str(rw):>12s}  {flag}")
        # Print params for the manipulation terms.
        np_ = native_tbl.get(t, (None, {}, None))[1]
        rp_ = rv_tbl.get(t, (None, {}, None))[1]
        if t in ("lift", "lift_precise", "joint_vel_hinge"):
            nstd = {k: v for k, v in np_.items() if k in ("reaching_std", "bringing_std", "std", "max_vel")}
            nsite = np_.get("asset_cfg", None)
            nsite = getattr(nsite, "site_names", None) if nsite is not None else None
            rstd = {k: v for k, v in rp_.items() if k in ("reaching_std", "bringing_std", "std", "max_vel")}
            rsite = rp_.get("site_name", None)
            print(f"    native params: {nstd} site={nsite}")
            print(f"    rv     params: {rstd} site={rsite}")
    return mismatches


# ---------------------------------------------------------------------------
# state injection
# ---------------------------------------------------------------------------
def _sample(rng):
    """Sample matched arm+finger joints, cube freejoint pose, target, action."""
    jpos = rng.uniform(-0.4, 0.4, size=N_JOINT).tolist()
    jvel = rng.uniform(-1.5, 1.5, size=N_JOINT).tolist()
    cube_pos = [rng.uniform(0.2, 0.45), rng.uniform(-0.2, 0.2), rng.uniform(0.04, 0.25)]
    yaw = rng.uniform(-math.pi, math.pi)
    cube_quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    target = [rng.uniform(0.3, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.4)]
    act = rng.uniform(-1.0, 1.0, size=7).tolist()
    return jpos, jvel, cube_pos, cube_quat, target, act


def _mj_cube_qposadr(mp):
    import mujoco

    bid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, "cube")
    jid = int(mp.body_jntadr[bid])
    return int(mp.jnt_qposadr[jid])


def mjlab_inject(mj, jpos, jvel, cube_pos, cube_quat, target, act):
    robot = mj.scene["robot"]
    cube = mj.scene["cube"]
    robot.write_joint_state_to_sim(
        torch.tensor([jpos], dtype=torch.float, device=DEVICE),
        torch.tensor([jvel], dtype=torch.float, device=DEVICE),
    )
    pose = torch.tensor([cube_pos + cube_quat], dtype=torch.float, device=DEVICE)
    cube.write_root_link_pose_to_sim(pose)
    cube.write_root_link_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float, device=DEVICE))
    mj.sim.forward()
    mj.action_manager.process_action(torch.tensor([act], dtype=torch.float, device=DEVICE))
    cmd = mj.command_manager.get_term("lift_height")
    cmd.target_pos[:] = torch.tensor([target], dtype=torch.float, device=DEVICE)


def rv_inject(rv, jpos, jvel, cube_pos, cube_quat, target, act):
    ph = rv.handler.physics
    mp = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
    qadr = _mj_cube_qposadr(mp)
    with ph.reset_context():
        ph.data.qpos[0:N_JOINT] = jpos
        ph.data.qvel[0:N_JOINT] = jvel
        ph.data.qpos[qadr : qadr + 3] = cube_pos
        ph.data.qpos[qadr + 3 : qadr + 7] = cube_quat
    rv._action = torch.tensor([act], dtype=torch.float, device=rv.device)
    mgr = rv.command_managers["lift_height"]
    mgr._target_pos[:] = torch.tensor([target], dtype=torch.float, device=rv.device)


# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(SEED)
    mj = build_mjlab(_yam_builder)
    rv = build_rv("mjlab.lift_cube_yam_v2")

    native_tbl = native_term_table(mj)
    rv_tbl = rv_term_table(rv)
    mism = compare_term_sets(native_tbl, rv_tbl)

    shared = [t for t in rv_tbl if t in native_tbl]
    per_term_max = {t: 0.0 for t in shared}

    for _ in range(K):
        jpos, jvel, cube_pos, cube_quat, target, act = _sample(rng)
        mjlab_inject(mj, jpos, jvel, cube_pos, cube_quat, target, act)
        rv_inject(rv, jpos, jvel, cube_pos, cube_quat, target, act)
        # action_rate_l2: copy native's action/prev_action so the formula sees
        # identical inputs (history would otherwise differ).
        rv._action = mj.action_manager.action.clone().to(rv.device)
        rv._prev_action = mj.action_manager.prev_action.clone().to(rv.device)
        states = rv.handler.get_states(mode="tensor")
        for t in shared:
            try:
                nv = native_tbl[t][2](mj, **native_tbl[t][1]).detach().cpu().numpy().reshape(-1)[0]
                rvv = rv_tbl[t][2](rv, states, **rv_tbl[t][1]).detach().cpu().numpy().reshape(-1)[0]
                per_term_max[t] = max(per_term_max[t], float(abs(nv - rvv)))
            except Exception as e:
                per_term_max[t] = f"ERR: {str(e)[:90]}"

    print("\n----- YAM: per-term max|Δ| (matched state) -----")
    ok = True
    for t, v in per_term_max.items():
        vs = f"{v:.3e}" if isinstance(v, float) else v
        verdict = "PASS(eps)" if (isinstance(v, float) and v <= TOL) else "FAIL/CHECK"
        if not (isinstance(v, float) and v <= TOL):
            ok = False
        print(f"   {t:26s} max|Δ|={vs:>12}  {verdict}")
    if mism:
        print(f"\n   term-set/weight mismatches: {mism}")
    print("\nYAM OVERALL:", "PASS" if ok else "FAIL")
    return ok


def main():
    try:
        ok = run()
    except Exception:
        print(f"\n!!! YAM FAILED:\n{traceback.format_exc()[-2500:]}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
