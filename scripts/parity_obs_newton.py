"""Newton-path actor-observation parity vs native mjlab.

Companion to scripts/parity_obs_all.py. That harness validates the metasim
*mujoco* port against native mjlab by injecting identical (root, jpos, jvel,
command, last_action) tuples via ``handler.physics`` and comparing the actor
observation. This script does the same for the *Newton* (SolverMuJoCo GPU)
backend: it reuses the native mjlab side and per-task samplers from
parity_obs_all, but builds the RoboVerse env with ``simulator="newton"`` and
injects the matched state through the standard ``handler.set_states`` API
(constructing a TensorState whose root_state == the same 13-vector and whose
joint_pos/joint_vel follow the same joint order, which parity_obs_all already
verified matches mjlab).

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
        MJLAB_G1_MOTION_FILE=/tmp/g1_synth_motion.npz \
        python scripts/parity_obs_newton.py
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "parity_obs_all", os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_obs_all.py")
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

DEVICE = "cuda:0"
JSON_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity_obs_newton.json")


def build_rv_newton(task_id, robot_name):
    import roboverse_pack  # noqa: F401
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.task.registry import get_task_class

    scn = ScenarioCfg(
        robots=[robot_name], objects=[], cameras=[],
        sim_params=SimParamCfg(dt=0.005), decimation=4,
        simulator="newton", num_envs=1, headless=True,
    )
    env = get_task_class(task_id)(scenario=scn, device=DEVICE)
    env.reset()
    return env


def _robot_key(env):
    ts = env.handler.get_states(mode="tensor")
    return next(iter(ts.robots.keys())), ts


def rv_inject_newton(env, robot_key, *, root, jpos, jvel, command, last_action, free_base, perm=None):
    """Inject matched state into the Newton handler via set_states + TensorState.

    ``perm`` reorders jpos/jvel from the source (mjlab) joint order into the
    Newton TensorState joint order (which is sorted by joint name).
    """
    ts = env.handler.get_states(mode="tensor")
    rs = ts.robots[robot_key]
    if free_base and root is not None:
        rs.root_state[:] = torch.tensor(root, dtype=torch.float, device=DEVICE).unsqueeze(0)
    jp = np.asarray(jpos, dtype=np.float32)
    jv = np.asarray(jvel, dtype=np.float32)
    if perm is not None:
        jp = jp[perm]
        jv = jv[perm]
    rs.joint_pos[:] = torch.tensor(jp, device=DEVICE).unsqueeze(0)
    rs.joint_vel[:] = torch.tensor(jv, device=DEVICE).unsqueeze(0)
    env.handler.set_states(ts)

    if last_action is not None:
        env._action = torch.tensor([last_action], dtype=torch.float, device=env.device)
    if command is not None:
        for cname, cvec in command.items():
            mgr = env.command_managers.get(cname)
            if mgr is not None and hasattr(mgr, "_command"):
                n = min(mgr._command.shape[-1], len(cvec))
                mgr._command[:, :n] = torch.tensor(cvec[:n], dtype=torch.float, device=env.device)


def _joint_perm(mj, ent_name, rv, rkey):
    """Permutation mapping mjlab joint order -> Newton TensorState (sorted) order."""
    mj_order = list(mj.scene[ent_name].joint_names)
    nt_order = rv.handler._get_joint_names(rkey, sort=True)
    return [mj_order.index(n) for n in nt_order]


def run_velocity_newton(name, native_builder, rv_task, robot_name, ent_name, n_joints, rng):
    mj = P.build_mjlab(native_builder)
    rv = build_rv_newton(rv_task, robot_name)
    rkey, _ = _robot_key(rv)
    perm = _joint_perm(mj, ent_name, rv, rkey)
    mj_layout = P.native_term_layout(mj)
    rv_layout = P.rv_term_layout(rv)
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    mo = ro = None
    for _ in range(P.K):
        root, jpos, jvel, cmd, act = P._sample_velocity(rng, n_joints)
        P.mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, ent_name=ent_name)
        rv_inject_newton(rv, rkey, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, free_base=True, perm=perm)
        mo = P.native_compute_actor(mj)
        ro = P.rv_compute_actor(rv)
        o, pt, st, nn = P.compare_term_aligned(mo, mj_layout, ro, rv_layout)
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
        task=name, obs_dim=len(mo) if mo is not None else None, rv_dim=len(ro) if ro is not None else None,
        max_dobs=overall if structural else None, structural=structural,
        per_term={k: v[0] for k, v in per_term_max.items()}, notes=notes,
    )


def run_cartpole_newton(name, native_builder, rv_task, robot_name, rng):
    mj = P.build_mjlab(native_builder)
    rv = build_rv_newton(rv_task, robot_name)
    rkey, _ = _robot_key(rv)
    perm = _joint_perm(mj, "cartpole", rv, rkey)
    mj_layout = P.native_term_layout(mj)
    rv_layout = P.rv_term_layout(rv)
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    mo = ro = None
    for _ in range(P.K):
        root, jpos, jvel, cmd, act = P._sample_cartpole(rng)
        P.mjlab_inject(mj, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="cartpole")
        rv_inject_newton(rv, rkey, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, free_base=False, perm=perm)
        mo = P.native_compute_actor(mj)
        ro = P.rv_compute_actor(rv)
        o, pt, st, nn = P.compare_term_aligned(mo, mj_layout, ro, rv_layout)
        structural = st
        if not st:
            notes = nn
            break
        overall = max(overall, o)
        for k2, (v, d) in pt.items():
            per_term_max[k2] = (max(per_term_max.get(k2, (0.0, d))[0], v), d)
    return dict(
        task=name, obs_dim=len(mo) if mo is not None else None, rv_dim=len(ro) if ro is not None else None,
        max_dobs=overall if structural else None, structural=structural,
        per_term={k: v[0] for k, v in per_term_max.items()}, notes=notes,
    )


def run_tracking_newton(name, robot_name, rng):
    """Tracking obs parity on Newton (g1, 29 joints, free base + motion command).

    Injects matched (root, jpos, jvel, last_action) into native mjlab + Newton
    RV and compares the actor obs term-by-term. The PROPRIOCEPTIVE terms
    (base_lin_vel/base_ang_vel/joint_pos/joint_vel/last_action) are fully
    determined by the injected state and should match at float-eps. The
    MOTION-FRAME terms (command / motion_anchor_pos_b / motion_anchor_ori_b)
    depend on the motion clip frame AND the robot's per-body world kinematics;
    those are reported per-term but not expected to match unless body
    kinematics are injected identically (same structural gap as the mujoco
    tracking port).
    """
    motion = os.environ.get("MJLAB_G1_MOTION_FILE")
    if not (motion and os.path.exists(motion)):
        return dict(task=name, error="BLOCKED: MJLAB_G1_MOTION_FILE not set/missing")
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    def builder(play=False):
        cfg = unitree_g1_flat_tracking_env_cfg(play=play)
        cfg.commands["motion"].motion_file = motion
        return cfg

    mj = P.build_mjlab(builder)
    rv = build_rv_newton(name, robot_name)
    rkey, _ = _robot_key(rv)
    perm = _joint_perm(mj, "robot", rv, rkey)
    mj_layout = P.native_term_layout(mj)
    rv_layout = P.rv_term_layout(rv)
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    mo = ro = None
    # Proprioceptive terms that ARE determined by the injected state.
    proprio = {"base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "last_action", "actions"}
    proprio_max = 0.0
    for _ in range(P.K):
        root, jpos, jvel, cmd, act = P._sample_velocity(rng, 29)
        P.mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="robot")
        rv_inject_newton(rv, rkey, root=root, jpos=jpos, jvel=jvel, command=None, last_action=act, free_base=True, perm=perm)
        mo = P.native_compute_actor(mj)
        ro = P.rv_compute_actor(rv)
        o, pt, st, nn = P.compare_term_aligned(mo, mj_layout, ro, rv_layout)
        structural = st
        if st is False:
            notes = [f"DIM/STRUCT MISMATCH native={mo.shape} rv={ro.shape}"] + nn
            break
        overall = max(overall, o)
        for k2, (v, d) in pt.items():
            per_term_max[k2] = (max(per_term_max.get(k2, (0.0, d))[0], v), d)
            if k2 in proprio:
                proprio_max = max(proprio_max, v)
    notes.append(
        "motion-frame terms (command/motion_anchor_*) depend on the motion clip "
        "+ robot body kinematics and are NOT injected identically; "
        f"proprioceptive-only max|Δ|={proprio_max:.3e}"
    )
    return dict(
        task=name, obs_dim=len(mo) if mo is not None else None, rv_dim=len(ro) if ro is not None else None,
        max_dobs=overall if structural else None, structural=structural,
        proprio_max=proprio_max,
        per_term={k: v[0] for k, v in per_term_max.items()}, notes=notes,
    )


def run_yam_newton(name, robot_name, rng):
    """YAM obs parity on Newton.

    Injects matched 8-joint (6 arm + 2 finger) jpos/jvel + last_action into
    native mjlab + Newton RV and compares the actor obs term-by-term. The
    joint-state terms are determined by the injection; the EE-site / cube /
    goal terms depend on the cube/goal pose (a scene object on Newton vs an
    MJCF body on mujoco) which is not injected identically, so those are
    reported per-term but excluded from the joint-state PASS criterion.
    """
    mj = P.build_mjlab(P._yam_builder)
    rv = build_rv_newton(name, robot_name)
    rkey, _ = _robot_key(rv)
    perm = _joint_perm(mj, "robot", rv, rkey)
    mj_layout = P.native_term_layout(mj)
    rv_layout = P.rv_term_layout(rv)
    overall = 0.0
    per_term_max = {}
    structural = None
    notes = []
    mo = ro = None
    jointstate = {"joint_pos", "joint_vel", "last_action", "actions"}
    js_max = 0.0
    for _ in range(P.K):
        root, jpos, jvel, cmd, act = P._sample_yam(rng)
        P.mjlab_inject(mj, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, ent_name="robot")
        rv_inject_newton(rv, rkey, root=None, jpos=jpos, jvel=jvel, command=None, last_action=act, free_base=False, perm=perm)
        mo = P.native_compute_actor(mj)
        ro = P.rv_compute_actor(rv)
        o, pt, st, nn = P.compare_term_aligned(mo, mj_layout, ro, rv_layout)
        structural = st
        if st is False:
            notes = [f"DIM/STRUCT MISMATCH native={mo.shape} rv={ro.shape}"] + nn
            break
        overall = max(overall, o)
        for k2, (v, d) in pt.items():
            per_term_max[k2] = (max(per_term_max.get(k2, (0.0, d))[0], v), d)
            if k2 in jointstate:
                js_max = max(js_max, v)
    notes.append(
        "EE-site/cube/goal terms depend on the cube+goal pose (scene object on "
        "Newton vs MJCF body on mujoco) and are NOT injected identically; "
        f"joint-state-only max|Δ|={js_max:.3e}"
    )
    return dict(
        task=name, obs_dim=len(mo) if mo is not None else None, rv_dim=len(ro) if ro is not None else None,
        max_dobs=overall if structural else None, structural=structural,
        jointstate_max=js_max,
        per_term={k: v[0] for k, v in per_term_max.items()}, notes=notes,
    )


def main():
    rng = np.random.default_rng(P.SEED)
    results = []
    specs = [
        ("cartpole", "mjlab.cartpole_balance_v2", "mjlab_cartpole", P._cartpole_balance_builder),
        ("cartpole", "mjlab.cartpole_swingup_v2", "mjlab_cartpole", P._cartpole_swingup_builder),
        ("vel", "mjlab.velocity_flat_go1_v2", "mjlab_go1", P._go1_flat_builder, "robot", 12),
        ("vel", "mjlab.velocity_rough_go1_v2", "mjlab_go1", P._go1_rough_builder, "robot", 12),
        ("vel", "mjlab.velocity_flat_g1_v2", "mjlab_g1", P._g1_flat_builder, "robot", 29),
        ("vel", "mjlab.velocity_rough_g1_v2", "mjlab_g1", P._g1_rough_builder, "robot", 29),
        ("track", "mjlab.tracking_flat_g1_v2", "mjlab_g1"),
        ("track", "mjlab.tracking_flat_g1_no_state_est_v2", "mjlab_g1"),
        ("yam", "mjlab.lift_cube_yam_v2", "mjlab_yam"),
        ("yam", "mjlab.lift_cube_yam_depth_v2", "mjlab_yam"),
        ("yam", "mjlab.lift_cube_yam_rgb_v2", "mjlab_yam"),
        ("yam", "mjlab.multi_cube_seg_yam_v2", "mjlab_yam"),
    ]
    for spec in specs:
        kind = spec[0]
        try:
            if kind == "cartpole":
                _, task, robot, builder = spec
                r = run_cartpole_newton(task, builder, task, robot, rng)
            elif kind == "vel":
                _, task, robot, builder, ent, nj = spec
                r = run_velocity_newton(task, builder, task, robot, ent, nj, rng)
            elif kind == "track":
                _, task, robot = spec
                r = run_tracking_newton(task, robot, rng)
            elif kind == "yam":
                _, task, robot = spec
                r = run_yam_newton(task, robot, rng)
        except Exception as e:
            import traceback

            traceback.print_exc()
            r = dict(task=spec[1], error=f"{type(e).__name__}: {str(e)[:120]}")
        results.append(r)
        md = r.get("max_dobs")
        md_s = "None" if md is None else f"{md:.3e}"
        extra = ""
        if r.get("proprio_max") is not None:
            extra = f" proprio={r['proprio_max']:.3e}"
        elif r.get("jointstate_max") is not None:
            extra = f" jointstate={r['jointstate_max']:.3e}"
        print(f"{r['task']:42s} max|dObs|={md_s:>12} struct={r.get('structural')}{extra} notes={r.get('notes') or r.get('error','')}")

    json.dump({"K": P.K, "results": results}, open(JSON_OUT, "w"), indent=1)
    print(f"\nwrote {JSON_OUT}")


if __name__ == "__main__":
    main()
