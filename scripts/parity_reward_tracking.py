"""Per-reward-term parity: native mjlab G1 tracking vs RoboVerse v2.

Builds BOTH the native mjlab G1 tracking env and the RoboVerse v2 tracking env
with the SAME synthetic motion clip, pins the SAME motion frame on both command
managers, injects K matched robot states (root + joint + last_action), and
compares each ``motion_*_error_exp`` reward term value at float-eps.

The MotionCommand quantities the rewards read (anchor pos/ori, relative body
pos/ori, body lin/ang vel + robot_* equivalents) are recomputed on both sides
from the matched physics state + pinned motion frame, so any divergence is a
formula bug (fixed in commands.py / rewards.py), not unmatched history.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 MJLAB_G1_MOTION_FILE=/tmp/g1_synth_motion.npz \
        python scripts/parity_reward_tracking.py
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
    build_mjlab,
    build_rv,
    mjlab_inject,
    rv_inject,
)

K = 24
SEED = 5
TOL = 1e-5
MOTION = os.environ.get("MJLAB_G1_MOTION_FILE", "/tmp/g1_synth_motion.npz")
N_JOINTS = 29

# native reward term name -> RV reward term name
TERM_ALIAS = {
    "motion_global_root_pos": "track_anchor_pos",
    "motion_global_root_ori": "track_anchor_orient",
    "motion_body_pos": "track_body_pos",
    "motion_body_ori": "track_body_orient",
    "motion_body_lin_vel": "track_body_lin_vel",
    "motion_body_ang_vel": "track_body_ang_vel",
}


def native_builder(play=False):
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)
    cfg.commands["motion"].motion_file = MOTION
    if "encoder_bias" in cfg.events:
        cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    return cfg


def native_reward_table(mj):
    rm = mj.reward_manager
    out = {}
    for name, cfg in zip(rm.active_terms, rm._term_cfgs):
        out[name] = (cfg.weight, cfg.params, cfg.func)
    return out


def rv_reward_table(rv):
    out = {}
    for name in rv._term_names(rv.cfg.rewards):
        term = getattr(rv.cfg.rewards, name)
        out[name] = (term.weight, term.params or {}, term.func)
    return out


def _sample(rng):
    root = [
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.55, 0.85),
        1.0, 0.0, 0.0, 0.0,
        rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
    ]
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    ang = rng.uniform(0, 0.6)
    root[3] = math.cos(ang / 2)
    root[4:7] = (math.sin(ang / 2) * ax).tolist()
    jpos = rng.uniform(-0.4, 0.4, size=N_JOINTS).tolist()
    jvel = rng.uniform(-1.0, 1.0, size=N_JOINTS).tolist()
    act = rng.uniform(-1.0, 1.0, size=N_JOINTS).tolist()
    return root, jpos, jvel, act


def _pin_frame(mj, rv, frame):
    mj.command_manager.get_term("motion").time_steps[:] = frame
    rv.command_managers["motion"].time_steps[:] = frame
    # native recomputes body_pos_relative_w / body_quat_relative_w via
    # update_relative_body_poses() each step; call it after pinning + state
    # injection so the cached relative tensors reflect the injected robot state.
    mj.command_manager.get_term("motion").update_relative_body_poses()


def compare_weights(native_tbl, rv_tbl):
    print("\n===== TRACKING: term/weight/std comparison (native vs RV) =====")
    print(f"{'native term':28s} {'rv term':22s} {'nat_w':>7s} {'rv_w':>7s} {'nat_std':>8s} {'rv_std':>8s}  flag")
    mismatches = []
    for nname, rname in TERM_ALIAS.items():
        nw, np_, _ = native_tbl[nname]
        rw, rp_, _ = rv_tbl[rname]
        nstd = np_.get("std")
        rstd = rp_.get("std")
        flag = ""
        if abs(float(nw) - float(rw)) > 1e-12:
            flag += "WEIGHT "
            mismatches.append((nname, "weight", nw, rw))
        if nstd is not None and rstd is not None and abs(float(nstd) - float(rstd)) > 1e-9:
            flag += "STD"
            mismatches.append((nname, "std", nstd, rstd))
        print(f"{nname:28s} {rname:22s} {nw:7.3g} {rw:7.3g} {str(nstd):>8s} {str(rstd):>8s}  {flag}")
    return mismatches


def run():
    assert os.path.exists(MOTION), f"motion file missing: {MOTION}"
    rng = np.random.default_rng(SEED)
    mj = build_mjlab(native_builder)
    rv = build_rv("mjlab.tracking_flat_g1_v2")

    native_tbl = native_reward_table(mj)
    rv_tbl = rv_reward_table(rv)
    mism = compare_weights(native_tbl, rv_tbl)

    T = int(mj.command_manager.get_term("motion").motion.time_step_total)
    per_term_max = {n: 0.0 for n in TERM_ALIAS}

    for _ in range(K):
        root, jpos, jvel, act = _sample(rng)
        frame = int(rng.integers(0, T))
        mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="robot")
        rv_inject(rv, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act,
                  n_qpos_joint=N_JOINTS, free_base=True)
        _pin_frame(mj, rv, frame)
        states = rv.handler.get_states(mode="tensor")
        for nname, rname in TERM_ALIAS.items():
            try:
                nv = native_tbl[nname][2](mj, **native_tbl[nname][1]).detach().cpu().numpy().reshape(-1)[0]
                rvv = rv_tbl[rname][2](rv, states, **rv_tbl[rname][1]).detach().cpu().numpy().reshape(-1)[0]
                per_term_max[nname] = max(per_term_max[nname], float(abs(nv - rvv)))
            except Exception as e:
                per_term_max[nname] = f"ERR: {str(e)[:90]}"

    print("\n----- TRACKING: per-term max|Δ| (matched state + pinned frame) -----")
    ok = True
    for nname in TERM_ALIAS:
        v = per_term_max[nname]
        vs = f"{v:.3e}" if isinstance(v, float) else v
        good = isinstance(v, float) and v <= TOL
        ok = ok and good
        print(f"   {nname:28s} ({TERM_ALIAS[nname]:22s}) max|Δ|={vs:>12}  {'PASS(eps)' if good else 'FAIL/CHECK'}")
    if mism:
        print(f"\n   term-set/weight/std mismatches: {mism}")
    print("\nTRACKING OVERALL:", "PASS" if ok else "FAIL")
    return ok


def main():
    try:
        ok = run()
    except Exception:
        print(f"\n!!! TRACKING FAILED:\n{traceback.format_exc()[-2500:]}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
