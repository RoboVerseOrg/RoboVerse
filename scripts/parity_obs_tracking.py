"""Tracking actor-observation parity: native mjlab vs RoboVerse v2.

Builds BOTH the native mjlab G1 tracking env and the RoboVerse v2 tracking env
with the SAME synthetic motion clip, injects K matched states (root + joint +
last_action), pins the SAME motion frame on both command managers, recomputes
the full ``actor`` observation, and compares term-by-term at float-eps. Done for
BOTH variants:

  mjlab.tracking_flat_g1_v2              (has_state_estimation=True,  160-D)
  mjlab.tracking_flat_g1_no_state_est_v2 (has_state_estimation=False, 154-D)

State injection reuses the helpers from ``scripts/parity_obs_all.py``.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 MJLAB_G1_MOTION_FILE=/tmp/g1_synth_motion.npz \
        python scripts/parity_obs_tracking.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

from scripts.parity_obs_all import (
    DEVICE,
    build_mjlab,
    build_rv,
    mjlab_inject,
    native_compute_actor,
    native_term_layout,
    rv_compute_actor,
    rv_inject,
)

K = 16
SEED = 0
TOL = 1e-5
MOTION = os.environ.get("MJLAB_G1_MOTION_FILE", "/tmp/g1_synth_motion.npz")
N_JOINTS = 29


def native_builder_factory(has_state_estimation: bool):
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    def builder(play=False):
        cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=has_state_estimation, play=play)
        cfg.commands["motion"].motion_file = MOTION
        # The RV port has no encoder-bias randomization, so the native
        # ``joint_pos_rel(biased=True)`` term would diverge by the random startup
        # bias. Zero the bias range so both sides see an unbiased joint_pos obs
        # (a parity-only setup; the bias is pure domain randomization).
        if "encoder_bias" in cfg.events:
            cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
        return cfg

    return builder


def _sample(rng):
    """Sample a matched root + joint + last_action state (G1 free base)."""
    import math

    root = [
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.55, 0.85),
        1.0,
        0.0,
        0.0,
        0.0,
        rng.uniform(-0.8, 0.8),
        rng.uniform(-0.8, 0.8),
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4),
    ]
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    ang = rng.uniform(0, 0.5)
    root[3] = math.cos(ang / 2)
    root[4:7] = (math.sin(ang / 2) * ax).tolist()
    jpos = rng.uniform(-0.4, 0.4, size=N_JOINTS).tolist()
    jvel = rng.uniform(-1.0, 1.0, size=N_JOINTS).tolist()
    act = rng.uniform(-1.0, 1.0, size=N_JOINTS).tolist()
    return root, jpos, jvel, act


def _pin_frame(mj, rv, frame: int) -> None:
    """Pin the SAME motion time step on both command managers."""
    mj_cmd = mj.command_manager.get_term("motion")
    mj_cmd.time_steps[:] = frame
    rv_cmd = rv.command_managers["motion"]
    rv_cmd.time_steps[:] = frame


def run_variant(name: str, has_state_estimation: bool, rng):
    mj = build_mjlab(native_builder_factory(has_state_estimation))
    rv = build_rv(name)

    mj_layout = native_term_layout(mj)
    rv_layout = native_term_layout  # placeholder; RV uses its own below
    from scripts.parity_obs_all import rv_term_layout

    rv_layout = rv_term_layout(rv)

    # Native names the final term ``actions``; the RV port calls the identical
    # term ``last_action`` (same alias the parity_obs_all harness uses).
    _alias = {"last_action": "actions"}

    def _norm(seq):
        return [_alias.get(n, n) for n, _ in seq]

    mj_names = _norm(mj_layout)
    rv_names = _norm(rv_layout)
    structural = mj_names == rv_names

    per_term_max: dict[str, tuple[float, int]] = {}
    overall = 0.0
    mo = ro = None
    T = int(mj.command_manager.get_term("motion").motion.time_step_total)

    for i in range(K):
        root, jpos, jvel, act = _sample(rng)
        frame = int(rng.integers(0, T))
        mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="robot")
        rv_inject(rv, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act,
                  n_qpos_joint=N_JOINTS, free_base=True)
        _pin_frame(mj, rv, frame)

        mo = native_compute_actor(mj)
        ro = rv_compute_actor(rv)

        if mo.shape != ro.shape:
            return dict(task=name, obs_dim=int(mo.shape[-1]), rv_dim=int(ro.shape[-1]),
                        max_dobs=None, structural=False, per_term={},
                        notes=[f"DIM MISMATCH native={mo.shape} rv={ro.shape}",
                               f"native={mj_layout}", f"rv={rv_layout}"])
        # term-by-term (positional; names verified structural)
        off = 0
        for (n, d) in mj_layout:
            md = float(np.abs(mo[off:off + d] - ro[off:off + d]).max()) if d else 0.0
            prev = per_term_max.get(n, (0.0, d))[0]
            per_term_max[n] = (max(prev, md), d)
            overall = max(overall, md)
            off += d

    notes = []
    if not structural:
        notes.append(f"NAME ORDER DIFF native={mj_names} rv={rv_names}")
    return dict(
        task=name,
        obs_dim=int(mo.shape[-1]) if mo is not None else None,
        rv_dim=int(ro.shape[-1]) if ro is not None else None,
        max_dobs=overall,
        structural=structural,
        per_term={k: v[0] for k, v in per_term_max.items()},
        per_term_dim={k: v[1] for k, v in per_term_max.items()},
        notes=notes,
    )


def _print(r):
    md = r.get("max_dobs")
    verdict = "PASS" if (r.get("structural") and md is not None and md <= TOL) else "FAIL"
    print(f"\n=== {r['task']} ===  native_dim={r['obs_dim']} rv_dim={r['rv_dim']} "
          f"structural={r['structural']} max|Δ|={md} -> {verdict}")
    print(f"  {'native term':24s} {'dim':>4s} {'max|Δ|':>12s}  status")
    for n, v in r.get("per_term", {}).items():
        d = r.get("per_term_dim", {}).get(n, "")
        st = "PASS" if v <= TOL else "FAIL"
        print(f"  {n:24s} {str(d):>4s} {v:12.3e}  {st}")
    for nn in r.get("notes", []):
        print(f"  note: {nn}")


def main():
    assert os.path.exists(MOTION), f"motion file missing: {MOTION} (run scripts/make_g1_motion.py)"
    rng = np.random.default_rng(SEED)
    results = []
    for name, hse in (("mjlab.tracking_flat_g1_v2", True),
                      ("mjlab.tracking_flat_g1_no_state_est_v2", False)):
        r = run_variant(name, hse, rng)
        results.append(r)
        _print(r)

    print("\n" + "=" * 72)
    ok = all(r.get("structural") and (r.get("max_dobs") is not None) and r["max_dobs"] <= TOL for r in results)
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
