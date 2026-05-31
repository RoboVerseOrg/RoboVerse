"""Full actor-observation parity harness: native mjlab vs RoboVerse v2 ports.

For each of the 12 mjlab tasks this builds BOTH the native mjlab env and the
RoboVerse v2 env, injects K=32 IDENTICAL matched states (root + joint state +
command + last_action) into each, recomputes the full ``actor`` observation
vector on both sides, and reports max|Δobs| overall and per native obs term.

Method (mirrors scripts/parity_obs_reward_cartpole.py + parity_go1_diag.py):
  - mjlab: ``entity.write_root_state_to_sim`` / ``write_joint_state_to_sim`` +
    ``sim.forward``; then invalidate ``observation_manager._obs_buffer=None``
    and call ``observation_manager.compute()``.
  - RoboVerse: write root + joint into ``handler.physics`` inside
    ``reset_context()``, then ``handler.get_states(mode='tensor')`` and
    ``env._observation_group('actor', states)``.
  - command: write the SAME vector into both command buffers.
  - last_action: write the SAME action buffer into both.

The native obs term order is read from the live mjlab ``observation_manager``;
the RoboVerse term order is read from the configclass field order. The harness
aligns the two by native term name and compares per-term. When the term SETS
or dims differ (a structural port gap, not a numeric one) it is reported
explicitly and the task is marked accordingly.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        python scripts/parity_obs_all.py
"""

from __future__ import annotations

import json
import math
import os
import traceback

import numpy as np
import torch

K = 32
SEED = 0
DEVICE = "cuda:0"
TOL = 1e-5
JSON_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity_obs_all.json")


# ---------------------------------------------------------------------------
# native mjlab helpers
# ---------------------------------------------------------------------------
def build_mjlab(builder):
    from mjlab.envs import ManagerBasedRlEnv

    cfg = builder(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=DEVICE)
    env.reset()
    return env


def native_term_layout(env, group="actor"):
    """Return [(name, dim), ...] for an obs group from the live manager."""
    om = env.observation_manager
    names = om._group_obs_term_names[group]
    dims = om._group_obs_term_dim[group]
    return [(n, int(np.prod(d))) for n, d in zip(names, dims)]


def native_compute_actor(env):
    env.observation_manager._obs_buffer = None
    return env.observation_manager.compute()["actor"][0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# RoboVerse helpers
# ---------------------------------------------------------------------------
def build_rv(task_id):
    import roboverse_pack  # noqa: F401
    from metasim.task.registry import get_task_class

    env = get_task_class(task_id)()
    env.reset()
    return env


def rv_term_layout(env, group="actor"):
    """Return [(field_name, dim), ...] for an RV obs group (configclass order)."""
    grp = getattr(env.cfg.observations, group)
    out = []
    states = env.handler.get_states(mode="tensor")
    for name in env._term_names(grp):
        term = getattr(grp, name)
        params = term.params or {}
        d = int(term.func(env, states, **params).shape[-1])
        out.append((name, d))
    return out


def rv_compute_actor(env):
    states = env.handler.get_states(mode="tensor")
    return env._observation_group("actor", states)[0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# state injection
# ---------------------------------------------------------------------------
def mjlab_inject(env, *, root, jpos, jvel, command, last_action, ent_name):
    ent = env.scene[ent_name]
    if root is not None:
        ent.write_root_state_to_sim(torch.tensor([root], dtype=torch.float, device=DEVICE))
    ent.write_joint_state_to_sim(
        torch.tensor([jpos], dtype=torch.float, device=DEVICE),
        torch.tensor([jvel], dtype=torch.float, device=DEVICE),
    )
    env.sim.forward()
    if last_action is not None:
        env.action_manager.process_action(torch.tensor([last_action], dtype=torch.float, device=DEVICE))
    if command is not None:
        for cname, cvec in command.items():
            term = env.command_manager.get_term(cname)
            buf = term.command  # property -> live buffer (vel_command_b etc.)
            n = min(buf.shape[-1], len(cvec))
            buf[:, :n] = torch.tensor(cvec[:n], dtype=torch.float, device=DEVICE)


def _world_to_body_angvel(quat_wxyz, ang_w):
    """Rotate a world-frame angular velocity into the free-joint body frame.

    MuJoCo's free-joint ``qvel[3:6]`` is the *body-frame* angular velocity,
    whereas mjlab's ``write_root_state_to_sim`` takes *world-frame* angular
    velocity and converts it (entity/data.py:114). To inject the SAME physical
    state into the RoboVerse physics.data.qvel we must apply the identical
    world->body rotation, else a tilted base receives a different ang vel on
    each side and base_ang_vel / base_lin_vel diverge spuriously.
    """
    w, x, y, z = quat_wxyz
    # conjugate rotation R(q)^T applied to ang_w (xyzw helper expects xyzw)
    q = np.array([x, y, z, w], dtype=np.float64)
    v = np.asarray(ang_w, dtype=np.float64)
    qv = q[:3]
    t = 2.0 * np.cross(-qv, v)
    return v + q[3] * t + np.cross(-qv, t)


def rv_inject(env, *, root, jpos, jvel, command, last_action, n_qpos_joint, free_base):
    ph = env.handler.physics
    with ph.reset_context():
        off = 0
        if free_base and root is not None:
            ph.data.qpos[0:3] = root[0:3]
            ph.data.qpos[3:7] = root[3:7]  # wxyz
            ph.data.qvel[0:3] = root[7:10]  # world-frame lin vel at body origin
            # qvel[3:6] is BODY-frame ang vel; root[10:13] is world-frame, so rotate.
            ph.data.qvel[3:6] = _world_to_body_angvel(root[3:7], root[10:13])
            off = 7
            voff = 6
        else:
            voff = 0
        ph.data.qpos[off : off + len(jpos)] = jpos
        ph.data.qvel[voff : voff + len(jvel)] = jvel
    if last_action is not None:
        env._action = torch.tensor([last_action], dtype=torch.float, device=env.device)
    if command is not None:
        for cname, cvec in command.items():
            mgr = env.command_managers.get(cname)
            if mgr is not None and hasattr(mgr, "_command"):
                n = min(mgr._command.shape[-1], len(cvec))
                mgr._command[:, :n] = torch.tensor(cvec[:n], dtype=torch.float, device=env.device)


# ---------------------------------------------------------------------------
# per-task specs
# ---------------------------------------------------------------------------
def _go1_flat_builder(play=False):
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    return unitree_go1_flat_env_cfg(play=play)


def _go1_rough_builder(play=False):
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg

    cfg = unitree_go1_rough_env_cfg(play=play)
    # num_envs=1 on rough terrain overflows the default contact buffer; raise it.
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 512
    return cfg


def _g1_flat_builder(play=False):
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    return unitree_g1_flat_env_cfg(play=play)


def _g1_rough_builder(play=False):
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

    return unitree_g1_rough_env_cfg(play=play)


def _cartpole_balance_builder(play=False):
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    return cartpole_balance_env_cfg(play=play)


def _cartpole_swingup_builder(play=False):
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_swingup_env_cfg

    return cartpole_swingup_env_cfg(play=play)


def _yam_builder(play=False):
    from mjlab.tasks.manipulation.config.yam.env_cfgs import yam_lift_cube_env_cfg

    return yam_lift_cube_env_cfg(play=play)


# Sampling ranges per family.
def _sample_velocity(rng, n_joints):
    root = [
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.30, 0.45),
        1.0,
        0.0,
        0.0,
        0.0,  # quat wxyz (kept upright-ish; randomize below)
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-0.5, 0.5),  # linvel world
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),  # angvel world
    ]
    # random small-tilt quaternion (wxyz)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-9
    ang = rng.uniform(0, 0.6)
    root[3] = math.cos(ang / 2)
    root[4:7] = (math.sin(ang / 2) * ax).tolist()
    jpos = rng.uniform(-0.4, 0.4, size=n_joints).tolist()
    jvel = rng.uniform(-1.0, 1.0, size=n_joints).tolist()
    cmd = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5)]
    act = rng.uniform(-1.0, 1.0, size=n_joints).tolist()
    return root, jpos, jvel, {"twist": cmd}, act


def _sample_cartpole(rng):
    jpos = [rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi)]
    jvel = [rng.uniform(-3, 3), rng.uniform(-6, 6)]
    act = [rng.uniform(-1.5, 1.5)]
    return None, jpos, jvel, None, act


def _sample_yam(rng):
    # native yam: 8 joints (6 arm + 2 fingers). action dim 7.
    jpos = rng.uniform(-0.5, 0.5, size=8).tolist()
    jvel = rng.uniform(-1.0, 1.0, size=8).tolist()
    act = rng.uniform(-1.0, 1.0, size=7).tolist()
    return None, jpos, jvel, None, act


# ---------------------------------------------------------------------------
# per-task compare driver
# ---------------------------------------------------------------------------
def compare_term_aligned(mj_obs, mj_layout, rv_obs, rv_layout):
    """Align by term name; return dict name->(max_abs_delta, dim) for shared
    terms plus overall, and a structural-match flag.

    Native term names are the canonical ones; RV term names may differ
    (e.g. 'last_action' vs 'actions', 'command' vs 'command'). We map by a
    normalized alias so position-by-position alignment holds when the term
    SEQUENCE matches.
    """
    alias = {
        "actions": "last_action",
        "last_action": "last_action",
        "command": "command",
        "pole_angle": "pole_angle",
    }

    def norm(seq):
        return [(alias.get(n, n), d) for n, d in seq]

    mj_n = norm(mj_layout)
    rv_n = norm(rv_layout)
    structural = mj_n == rv_n
    # RV layout is a strict prefix of native (native has extra trailing terms,
    # e.g. rough's height_scan that the RV port does not yet wire). We can still
    # compare the shared prefix term-by-term at float-eps.
    prefix = (not structural) and (len(rv_n) < len(mj_n)) and (mj_n[: len(rv_n)] == rv_n)
    per_term = {}
    notes = []
    if structural or prefix:
        common = rv_layout if prefix else mj_layout
        mo = 0
        ro = 0
        overall = 0.0
        for (name, d), (_, _) in zip(common, rv_layout):
            md = np.abs(mj_obs[mo : mo + d] - rv_obs[ro : ro + d])
            per_term[name] = (float(md.max()) if d else 0.0, d)
            overall = max(overall, per_term[name][0])
            mo += d
            ro += d
        if prefix:
            missing = mj_layout[len(rv_layout) :]
            notes.append(f"PREFIX-ONLY: RV missing native trailing terms {missing}")
        return overall, per_term, (True if structural else "prefix"), notes
    # structural mismatch: report what each side has
    notes.append(f"native_terms={mj_layout}")
    notes.append(f"rv_terms={rv_layout}")
    return None, per_term, False, notes


def run_velocity_task(name, native_builder, rv_task, ent_name, n_joints, rng):
    mj = build_mjlab(native_builder)
    rv = build_rv(rv_task)
    mj_layout = native_term_layout(mj)
    rv_layout = rv_term_layout(rv)
    mj_joints = list(mj.scene[ent_name].joint_names)
    # RV joint order: read from the obs cfg's joint asset_cfg
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    for i in range(K):
        root, jpos, jvel, cmd, act = _sample_velocity(rng, n_joints)
        mjlab_inject(
            mj, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, ent_name=ent_name
        )
        rv_inject(
            rv, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act,
            n_qpos_joint=n_joints, free_base=True,
        )
        mo = native_compute_actor(mj)
        ro = rv_compute_actor(rv)
        o, pt, st, nn = compare_term_aligned(mo, mj_layout, ro, rv_layout)
        structural = st
        if st is False:
            notes = [f"DIM/STRUCT MISMATCH native={mo.shape} rv={ro.shape}"] + nn
            break
        if nn:
            notes = nn
        overall = max(overall, o)
        for k2, (v, d) in pt.items():
            per_term_max[k2] = (max(per_term_max.get(k2, (0.0, d))[0], v), d)
    return dict(
        task=name,
        obs_dim=len(mo) if "mo" in dir() else None,
        rv_dim=len(ro) if "ro" in dir() else None,
        max_dobs=overall if structural else None,
        structural=structural,
        per_term={k: v[0] for k, v in per_term_max.items()},
        joint_order_match=(mj_joints == _rv_joint_order(rv_task)),
        notes=notes,
    )


def _rv_joint_order(rv_task):
    """Pull the RV joint name order used in obs for a given velocity task."""
    if "go1" in rv_task:
        from roboverse_pack.tasks.mjlab.velocity_go1_v2 import _GO1_JOINTS

        return list(_GO1_JOINTS.joint_names)
    if "g1" in rv_task:
        from roboverse_pack.tasks.mjlab.velocity_g1_v2 import _G1_ALL_JOINT_NAMES

        return list(_G1_ALL_JOINT_NAMES)
    return None


def run_cartpole_task(name, native_builder, rv_task, rng):
    mj = build_mjlab(native_builder)
    rv = build_rv(rv_task)
    mj_layout = native_term_layout(mj)
    rv_layout = rv_term_layout(rv)
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    mo = ro = None
    for i in range(K):
        root, jpos, jvel, cmd, act = _sample_cartpole(rng)
        mjlab_inject(mj, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="cartpole")
        rv_inject(rv, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, n_qpos_joint=2, free_base=False)
        mo = native_compute_actor(mj)
        ro = rv_compute_actor(rv)
        o, pt, st, nn = compare_term_aligned(mo, mj_layout, ro, rv_layout)
        structural = st
        if not st:
            notes = nn
            break
        overall = max(overall, o)
        for k2, (v, d) in pt.items():
            per_term_max[k2] = (max(per_term_max.get(k2, (0.0, d))[0], v), d)
    return dict(
        task=name,
        obs_dim=len(mo) if mo is not None else None,
        rv_dim=len(ro) if ro is not None else None,
        max_dobs=overall if structural else None,
        structural=structural,
        per_term={k: v[0] for k, v in per_term_max.items()},
        joint_order_match=True,
        notes=notes,
    )


def run_yam_task(name, native_builder, rv_task, rng):
    mj = build_mjlab(native_builder)
    rv = build_rv(rv_task)
    mj_layout = native_term_layout(mj)
    rv_layout = rv_term_layout(rv)
    notes = [
        f"native_actor_terms={mj_layout}",
        f"rv_actor_terms={rv_layout}",
    ]
    mo = native_compute_actor(mj)
    ro = rv_compute_actor(rv)
    structural = (mo.shape == ro.shape) and ([n for n, _ in mj_layout] == [
        {"actions": "actions"}.get(n, n) for n, _ in rv_layout
    ])
    return dict(
        task=name,
        obs_dim=int(mo.shape[-1]),
        rv_dim=int(ro.shape[-1]),
        max_dobs=None,
        structural=structural,
        per_term={},
        joint_order_match=None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    results = []

    tasks = [
        ("cart", "cartpole",
         lambda: run_cartpole_task("mjlab.cartpole_balance_v2", _cartpole_balance_builder, "mjlab.cartpole_balance_v2", rng)),
        ("cart", "cartpole",
         lambda: run_cartpole_task("mjlab.cartpole_swingup_v2", _cartpole_swingup_builder, "mjlab.cartpole_swingup_v2", rng)),
        ("vel", "go1",
         lambda: run_velocity_task("mjlab.velocity_flat_go1_v2", _go1_flat_builder, "mjlab.velocity_flat_go1_v2", "robot", 12, rng)),
        ("vel", "go1",
         lambda: run_velocity_task("mjlab.velocity_rough_go1_v2", _go1_rough_builder, "mjlab.velocity_rough_go1_v2", "robot", 12, rng)),
        ("vel", "g1",
         lambda: run_velocity_task("mjlab.velocity_flat_g1_v2", _g1_flat_builder, "mjlab.velocity_flat_g1_v2", "robot", 29, rng)),
        ("vel", "g1",
         lambda: run_velocity_task("mjlab.velocity_rough_g1_v2", _g1_rough_builder, "mjlab.velocity_rough_g1_v2", "robot", 29, rng)),
        ("yam", "yam",
         lambda: run_yam_task("mjlab.lift_cube_yam_v2", _yam_builder, "mjlab.lift_cube_yam_v2", rng)),
        ("yam", "yam",
         lambda: run_yam_task("mjlab.lift_cube_yam_depth_v2", _yam_builder, "mjlab.lift_cube_yam_depth_v2", rng)),
        ("yam", "yam",
         lambda: run_yam_task("mjlab.lift_cube_yam_rgb_v2", _yam_builder, "mjlab.lift_cube_yam_rgb_v2", rng)),
        ("yam", "yam",
         lambda: run_yam_task("mjlab.multi_cube_seg_yam_v2", _yam_builder, "mjlab.multi_cube_seg_yam_v2", rng)),
    ]

    for fam, _, fn in tasks:
        try:
            r = fn()
        except Exception as e:
            r = dict(task=str(e)[:60], error=traceback.format_exc()[-1500:], structural=None,
                     max_dobs=None, obs_dim=None, rv_dim=None, per_term={}, notes=["BUILD/RUN ERROR"])
        results.append(r)
        _print_row(r)

    # tracking tasks (need motion file).
    for tt in _tracking_tasks():
        try:
            r = tt(rng)
        except Exception as e:
            r = dict(task="tracking-error", error=traceback.format_exc()[-1500:], structural=None,
                     max_dobs=None, obs_dim=None, rv_dim=None, per_term={}, notes=[str(e)[:120]])
        results.append(r)
        _print_row(r)

    _summary(results)
    with open(JSON_OUT, "w") as f:
        json.dump({"K": K, "tol": TOL, "results": results}, f, indent=2, default=str)
    print(f"\nwrote {JSON_OUT}")


def _tracking_tasks():
    """Return tracking comparators. Tracking needs a motion npz fed to BOTH
    native and RV identically. We generate a synthetic motion below."""

    def make(name, no_state_est):
        def fn(rng):
            return run_tracking_task(name, no_state_est, rng)

        return fn

    return [
        make("mjlab.tracking_flat_g1_v2", False),
        make("mjlab.tracking_flat_g1_no_state_est_v2", True),
    ]


def run_tracking_task(name, no_state_est, rng):
    """Tracking obs parity needs a shared motion file + matched body kinematics.

    We attempt to build the native tracking env (requires a motion npz). If no
    motion file is available we report the structural layout and the blocker.
    """
    motion = os.environ.get("MJLAB_G1_MOTION_FILE")
    if not (motion and os.path.exists(motion)):
        return dict(
            task=name,
            obs_dim=None,
            rv_dim=None,
            max_dobs=None,
            structural=False,
            per_term={},
            joint_order_match=None,
            notes=[
                "BLOCKED: native tracking env requires a motion npz "
                "(MotionCommandCfg.motion_file). mjlab sources it from a W&B "
                "'motions' artifact at eval time; none is bundled in the repo. "
                "Set MJLAB_G1_MOTION_FILE to a MotionLoader-schema npz to compare. "
                "Also note RV tracking actor obs structurally differs from native "
                "(native = command, motion_anchor_pos_b, motion_anchor_ori_b, "
                "base_lin_vel, base_ang_vel, joint_pos, joint_vel, actions; "
                "RV reuses the velocity obs).",
            ],
        )
    # If a motion file is provided, build both and compare the non-motion terms.
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    def builder(play=False):
        cfg = unitree_g1_flat_tracking_env_cfg(play=play)
        cfg.commands["motion"].motion_file = motion
        return cfg

    mj = build_mjlab(builder)
    rv = build_rv(name)
    mj_layout = native_term_layout(mj)
    rv_layout = rv_term_layout(rv)
    return dict(
        task=name,
        obs_dim=sum(d for _, d in mj_layout),
        rv_dim=sum(d for _, d in rv_layout),
        max_dobs=None,
        structural=([n for n, _ in mj_layout] == [n for n, _ in rv_layout]),
        per_term={},
        joint_order_match=None,
        notes=[f"native={mj_layout}", f"rv={rv_layout}",
               "motion-frame terms (command/anchor) need identical motion + body kinematics; "
               "comparison limited to layout unless full kinematic injection is added."],
    )


def _verdict(r):
    md = r.get("max_dobs")
    st = r.get("structural")
    if st is True and md is not None and md <= TOL:
        return "PASS"
    if st == "prefix" and md is not None and md <= TOL:
        return "PASS(prefix)"
    if st is False:
        return "STRUCT-MISMATCH"
    if md is not None and md > TOL:
        return "FAIL"
    return "N/A"


def _print_row(r):
    status = _verdict(r)
    print(f"\n--- {r.get('task')} ---")
    print(f"   obs_dim(native)={r.get('obs_dim')} rv_dim={r.get('rv_dim')} "
          f"max|Δ|={r.get('max_dobs')} structural_match={r.get('structural')} "
          f"joint_order_match={r.get('joint_order_match')} -> {status}")
    if r.get("per_term"):
        for k, v in r["per_term"].items():
            print(f"      term {k:22s} max|Δ|={v:.3e}")
    for n in r.get("notes", []):
        print(f"      note: {n}")
    if r.get("error"):
        print(f"      ERROR:\n{r['error']}")


def _summary(results):
    print("\n" + "=" * 78)
    print(f"{'task':40s} {'ndim':>6s} {'max|Δobs|':>12s}  verdict")
    print("-" * 78)
    for r in results:
        md = r.get("max_dobs")
        v = _verdict(r)
        mds = f"{md:.3e}" if md is not None else "-"
        print(f"{str(r.get('task'))[:40]:40s} {str(r.get('obs_dim')):>6s} {mds:>12s}  {v}")


if __name__ == "__main__":
    main()
