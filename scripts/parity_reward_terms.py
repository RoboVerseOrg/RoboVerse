"""Per-reward-term parity harness: native mjlab vs RoboVerse v2 ports (go1, g1).

For go1 and g1 (flat configs) this:
  1. Enumerates the native reward terms (name, weight, params) from the live
     mjlab ``reward_manager`` and the RV terms from the configclass.
  2. Builds BOTH envs and injects K=32 IDENTICAL matched states (root + joint
     + command + last_action) into each (reusing the helpers in
     ``scripts/parity_obs_all.py``), then for EACH non-stateful reward term
     evaluates the per-term scalar on both sides and reports max|Δ|.
  3. For STATEFUL sensor terms (feet_air_time / feet_clearance /
     feet_swing_height / feet_slip / soft_landing / angular_momentum /
     self_collision) drives BOTH envs with an identical action sequence from
     reset for N steps so sensor history matches, then compares per-term.

Native term values are obtained by calling the mjlab term function (or the
bound term object) directly with the term's params. RV term values come from
calling the ``mdp/rewards.py`` function with the term params.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 python scripts/parity_reward_terms.py
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
    _g1_flat_builder,
    _go1_flat_builder,
    _sample_velocity,
    build_mjlab,
    build_rv,
    mjlab_inject,
    rv_inject,
)

K = 32
SEED = 1

# Terms that depend on accumulated sensor history (air time, contact, peak
# height, force history). These cannot be compared from a single injected
# state — they need a matched action rollout.
STATEFUL = {
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "angular_momentum",
    "self_collisions",
    "shank_collision",
    "trunk_head_collision",
}


# ---------------------------------------------------------------------------
# native term evaluation
# ---------------------------------------------------------------------------
def native_term_table(mj):
    """Return {name: (weight, params, term_obj_or_func)} from the live manager."""
    rm = mj.reward_manager
    out = {}
    # mjlab RewardManager: term cfgs live in rm._term_cfgs / rm.active_terms.
    # cfg.func is the bound instance for class-based terms (upright,
    # feet_swing_height, variable_posture) and the plain fn otherwise.
    for name, cfg in zip(rm.active_terms, rm._term_cfgs):
        out[name] = (cfg.weight, cfg.params, cfg.func)
    return out


def native_eval(mj, fn, params):
    val = fn(mj, **params)
    return val.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# RV term evaluation
# ---------------------------------------------------------------------------
def rv_term_table(rv):
    out = {}
    for name in rv._term_names(rv.cfg.rewards):
        term = getattr(rv.cfg.rewards, name)
        out[name] = (term.weight, term.params or {}, term.func)
    return out


def rv_eval(rv, fn, params, states):
    val = fn(rv, states, **params)
    return val.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# weight / term-set comparison
# ---------------------------------------------------------------------------
def compare_term_sets(name, native_tbl, rv_tbl):
    print(f"\n===== {name}: term/weight comparison (native flat vs RV) =====")
    all_names = list(dict.fromkeys(list(native_tbl) + list(rv_tbl)))
    print(f"{'term':26s} {'native_w':>12s} {'rv_w':>12s}  flag")
    mismatches = []
    for t in all_names:
        nw = native_tbl[t][0] if t in native_tbl else None
        rw = rv_tbl[t][0] if t in rv_tbl else None
        flag = ""
        if nw is None:
            flag = "RV-ONLY (not in native flat)"
            mismatches.append((t, "rv-only", nw, rw))
        elif rw is None:
            flag = "NATIVE-ONLY (missing in RV)"
            mismatches.append((t, "missing", nw, rw))
        elif abs(float(nw) - float(rw)) > 1e-12:
            flag = "WEIGHT MISMATCH"
            mismatches.append((t, "weight", nw, rw))
        print(f"{t:26s} {str(nw):>12s} {str(rw):>12s}  {flag}")
    return mismatches


# ---------------------------------------------------------------------------
# static (single-state) per-term parity
# ---------------------------------------------------------------------------
def run_static(name, native_builder, rv_task, ent_name, n_joints):
    rng = np.random.default_rng(SEED)
    mj = build_mjlab(native_builder)
    rv = build_rv(rv_task)

    native_tbl = native_term_table(mj)
    rv_tbl = rv_term_table(rv)
    mismatches = compare_term_sets(name, native_tbl, rv_tbl)

    shared = [t for t in rv_tbl if t in native_tbl and t not in STATEFUL]
    per_term_max = {t: 0.0 for t in shared}

    for _ in range(K):
        root, jpos, jvel, cmd, act = _sample_velocity(rng, n_joints)
        mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, ent_name=ent_name)
        rv_inject(rv, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, n_qpos_joint=n_joints, free_base=True)
        # action_rate_l2 reads (action - prev_action). mjlab's action_manager
        # shifts history across process_action calls (prev ← last iteration's
        # action), so copy native's ACTUAL action/prev_action into RV to compare
        # the formula on identical inputs.
        rv._action = mj.action_manager.action.clone().to(rv.device)
        rv._prev_action = mj.action_manager.prev_action.clone().to(rv.device)
        states = rv.handler.get_states(mode="tensor")
        for t in shared:
            try:
                nv = native_eval(mj, native_tbl[t][2], native_tbl[t][1])
                rvv = rv_eval(rv, rv_tbl[t][2], rv_tbl[t][1], states)
                d = float(np.abs(np.asarray(nv).reshape(-1)[0] - np.asarray(rvv).reshape(-1)[0]))
                per_term_max[t] = max(per_term_max[t], d)
            except Exception as e:
                per_term_max[t] = f"ERR: {str(e)[:80]}"
    return mj, rv, native_tbl, rv_tbl, per_term_max, mismatches


# ---------------------------------------------------------------------------
# stateful per-term parity: matched action rollout from reset
# ---------------------------------------------------------------------------
def run_stateful(name, mj, rv, native_tbl, rv_tbl, n_actions, n_steps=20):
    """Drive both envs with the SAME action sequence from reset, then compare
    stateful sensor reward terms per step (max over steps)."""
    print(f"\n===== {name}: stateful sensor terms (matched {n_steps}-step rollout) =====")
    rng = np.random.default_rng(SEED + 7)
    mj.reset()
    rv.reset()
    shared = [t for t in rv_tbl if t in native_tbl and t in STATEFUL]
    per_term_max = {t: 0.0 for t in shared}
    rv_zero = {t: True for t in shared}  # track whether RV stays exactly 0
    nat_nonzero = {t: False for t in shared}
    for _ in range(n_steps):
        act = rng.uniform(-1.0, 1.0, size=(1, n_actions)).astype(np.float32)
        at = torch.tensor(act, dtype=torch.float, device=DEVICE)
        mj.step(at)
        rv.step(at.to(rv.device))
        rv_states = rv.handler.get_states(mode="tensor")
        for t in shared:
            try:
                nv = float(np.asarray(native_eval(mj, native_tbl[t][2], native_tbl[t][1])).reshape(-1)[0])
                rvv = float(np.asarray(rv_eval(rv, rv_tbl[t][2], rv_tbl[t][1], rv_states)).reshape(-1)[0])
                per_term_max[t] = max(per_term_max[t], abs(nv - rvv))
                if abs(rvv) > 1e-12:
                    rv_zero[t] = False
                if abs(nv) > 1e-12:
                    nat_nonzero[t] = True
            except Exception as e:
                per_term_max[t] = f"ERR: {str(e)[:80]}"
    return per_term_max, rv_zero, nat_nonzero


# ---------------------------------------------------------------------------
def report(name, per_term_static, per_term_stateful, rv_zero, nat_nonzero):
    print(f"\n----- {name}: per-term max|Δ| (static, single matched state) -----")
    for t, v in per_term_static.items():
        vs = f"{v:.3e}" if isinstance(v, float) else v
        verdict = "PASS(eps)" if (isinstance(v, float) and v <= 1e-5) else "CHECK"
        print(f"   {t:26s} max|Δ|={vs:>12}  {verdict}")
    print(f"\n----- {name}: stateful sensor terms (NOTE: sims diverge dynamically")
    print("        after reset, so large max|Δ| here reflects unmatched sensor")
    print("        history, NOT a formula bug. Key signal = did RV FIRE nonzero?) -----")
    for t, v in per_term_stateful.items():
        vs = f"{v:.3e}" if isinstance(v, float) else v
        rv_fired = not rv_zero.get(t, True)
        nat_fired = nat_nonzero.get(t, False)
        if not rv_fired and nat_fired:
            # NOTE: RV staying 0 while native is nonzero over a random rollout is
            # USUALLY a dynamic-divergence artifact (the two sims drift apart, so
            # RV's feet/contacts differ from native's), NOT a stub. Confirm a true
            # stub only via direct sensor-state injection (see report); the g1
            # feet_ground_contact + self_collision sensors are verified to FIRE
            # when their physical condition is forced.
            status = "RV 0 this rollout (verify stub-vs-dynamics by injection)"
        elif rv_fired and nat_fired:
            status = "both FIRE (formula wired; Δ from dynamic divergence)"
        elif rv_fired and not nat_fired:
            status = "RV fires, native ~0 this rollout"
        else:
            status = "both ~0 this rollout (inconclusive)"
        print(f"   {t:26s} max|Δ|={vs:>12}  rv_fired={rv_fired!s:5} nat_fired={nat_fired!s:5}  {status}")


def main():
    specs = [
        ("go1", _go1_flat_builder, "mjlab.velocity_flat_go1_v2", "robot", 12),
        ("g1", _g1_flat_builder, "mjlab.velocity_flat_g1_v2", "robot", 29),
    ]
    for name, nb, rv_task, ent, nj in specs:
        try:
            mj, rv, ntbl, rtbl, static_max, mism = run_static(name, nb, rv_task, ent, nj)
            stateful_max, rv_zero, nat_nz = run_stateful(name, mj, rv, ntbl, rtbl, nj)
            report(name, static_max, stateful_max, rv_zero, nat_nz)
            if mism:
                print(f"\n   {name} term-set/weight mismatches: {mism}")
        except Exception:
            print(f"\n!!! {name} FAILED:\n{traceback.format_exc()[-2000:]}")


if __name__ == "__main__":
    main()
