"""Collect one successful RoboTwin episode as a sim-agnostic bimanual trajectory.

This is the *RoboTwin-side* half of the RoboVerse data bridge. It must run in
the dedicated ``robotwin`` conda env (sapien 3.0.0b1 / mplib / curobo), NOT the
``roboverse`` env -- their sapien/torch builds conflict. It drives a native
RoboTwin task through the same passthrough factory RoboVerse exposes
(``roboverse_pack.tasks.robotwin._passthrough``), retries seeds until one plans
*and* checks successfully (mirroring RoboTwin's ``collect_data.py`` loop), then
dumps the dense dual-arm joint trajectory plus initial object poses to a plain
pickle.

The companion ``get_started/10_robotwin_aloha_replay.py`` then runs in the
``roboverse`` env: it converts that pickle into RoboVerse's name-keyed ``*_v2``
format and replays the ALOHA-AgileX embodiment. Keeping the hand-off a plain
pickle of numpy arrays is what lets the two incompatible envs cooperate.

Example::

    conda run -n robotwin env MUJOCO_GL=egl python \\
        tools/robotwin_integration/collect_bridge.py --task beat_block_hammer \\
        --out ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import sys

import numpy as np

# The passthrough lives in the RoboVerse repo; load it by file path so this
# script does not import the full (heavy) ``roboverse_pack`` package in the
# robotwin env. ``_make_robotwin_env`` handles CWD + the warp.torch shim itself.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PASSTHROUGH = os.path.join(_REPO_ROOT, "roboverse_pack", "tasks", "robotwin", "_passthrough.py")


def _load_passthrough():
    spec = importlib.util.spec_from_file_location("rt_passthrough", _PASSTHROUGH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_capture_real_state(env) -> None:
    """Make every saved frame also carry RoboTwin's *achieved* arm qpos.

    RoboTwin's ``joint_action.vector`` is the PD *command target*
    (``joint.get_drive_target()``), not the physically achieved joint state, so
    comparing a RoboVerse replay against it would be circular (target-vs-target).
    The achieved qpos lives in ``robot.get_left/right_arm_real_jointState()``
    (``entity.get_qpos()``) but is not written to RoboTwin's per-frame cache. We
    wrap ``env.get_obs`` (the single hook ``_take_picture`` saves) to inject a
    ``joint_action.real_vector`` = achieved [L_arm(6), L_grip, R_arm(6), R_grip],
    aligned frame-for-frame with ``vector`` — a runtime-only shim, no upstream edit.
    """
    orig_get_obs = env.get_obs
    robot = env.robot
    scene = env.scene
    # The robot is itself an articulation; exclude it from the object capture.
    robot_arts = set()
    for attr in ("left_entity", "right_entity"):
        try:
            ent = getattr(robot, attr, None)
            if ent is not None:
                robot_arts.add(ent.get_name())
        except Exception:
            pass

    def get_obs_with_real():
        d = orig_get_obs()
        try:
            real = list(robot.get_left_arm_real_jointState()) + list(robot.get_right_arm_real_jointState())
            d.setdefault("joint_action", {})["real_vector"] = np.asarray(real, dtype=float)
        except Exception:
            pass  # achieved-state capture is best-effort; target vector is always present
        try:
            # Per-frame world pose of every scene object (rigid actors AND URDF
            # articulations, e.g. the pot/cabinet/laptop) minus the robot itself.
            poses = {}
            for actor in scene.get_all_actors():
                p = actor.get_pose()
                poses[actor.get_name()] = np.asarray([*p.p, *p.q], dtype=float)  # [x,y,z, qw,qx,qy,qz]
            for art in scene.get_all_articulations():
                name = art.get_name()
                if name in robot_arts:
                    continue
                p = art.get_root_pose() if hasattr(art, "get_root_pose") else art.get_pose()
                poses[name] = np.asarray([*p.p, *p.q], dtype=float)
            d["object_poses"] = poses
        except Exception:
            pass
        return d

    env.get_obs = get_obs_with_real


# Infrastructure actors (not manipulated targets) we never need a mesh for.
_INFRA_ACTORS = {"ground", "wall", "table", "table_wall"}


def _install_mesh_hook(pt) -> dict:
    """Record the exact mesh file RoboTwin loads for each named actor.

    ``create_actor`` resolves an object's mesh via the module-internal
    ``get_glb_or_obj_file(modeldir, model_id)`` and builds the actor with
    ``name=modelname`` (= the asset dir name). We wrap that resolver to record
    ``{modelname: {visual, scale}}`` so the RoboVerse replay can load the *real*
    mesh (not a primitive proxy) at the recorded pose. The runtime must be
    prepared (cwd + sys.path) before importing the RoboTwin envs package; the
    hook is module-level, so it persists across the per-seed retries.
    """
    import json as _json
    import os as _os

    rt = pt.robotwin_dir()
    pt._prepare_robotwin_runtime(rt)
    import sys as _sys

    if rt not in _sys.path:
        _sys.path.insert(0, rt)
    sink: dict = {}
    try:
        # ``envs.utils`` re-exports ``create_actor`` (the function), shadowing the
        # submodule attribute, so ``import envs.utils.create_actor as cau`` binds
        # the function. Fetch the actual module object from sys.modules instead.
        import importlib

        cau = importlib.import_module("envs.utils.create_actor")
    except Exception as e:
        print(f"[mesh-hook] could not import create_actor ({type(e).__name__}: {e}); meshes will be located by scan")
        return sink

    orig = cau.get_glb_or_obj_file

    def hooked(modeldir, model_id):
        f = orig(modeldir, model_id)
        try:
            name = _os.path.basename(str(modeldir).rstrip("/"))
            if name in ("visual", "collision"):
                name = _os.path.basename(_os.path.dirname(str(modeldir).rstrip("/")))
            scale = None
            jd = _os.path.join(rt, "assets", "objects", name)
            jf = _os.path.join(jd, "model_data.json" if model_id is None else f"model_data{model_id}.json")
            if _os.path.exists(jf):
                scale = _json.load(open(jf)).get("scale")
            # Record the mesh path relative to the RoboTwin checkout (portable).
            # create_actor resolves collision first then visual; prefer the visual
            # (render) mesh for a faithful replay, but keep whatever we got otherwise.
            rel = _os.path.relpath(str(f), rt)
            is_visual = f"{_os.sep}visual{_os.sep}" in str(f)
            if name not in sink or is_visual:
                sink[name] = {"visual": rel, "scale": scale, "model_id": model_id}
        except Exception:
            pass
        return f

    cau.get_glb_or_obj_file = hooked
    return sink


def _capture_objects(env) -> dict:
    """Record every scene object's initial world pose (rigid actors + articulations)."""
    objs: dict = {}
    robot_arts = set()
    for attr in ("left_entity", "right_entity"):
        try:
            ent = getattr(env.robot, attr, None)
            if ent is not None:
                robot_arts.add(ent.get_name())
        except Exception:
            pass
    try:
        actors = list(env.scene.get_all_actors())
    except Exception:
        actors = []
    for actor in actors:
        try:
            pose = actor.get_pose()
            objs[actor.get_name()] = {"pos": list(map(float, pose.p)), "rot": list(map(float, pose.q))}
        except Exception:
            continue
    try:
        arts = list(env.scene.get_all_articulations())
    except Exception:
        arts = []
    for art in arts:
        try:
            if art.get_name() in robot_arts:
                continue
            pose = art.get_root_pose() if hasattr(art, "get_root_pose") else art.get_pose()
            objs[art.get_name()] = {
                "pos": list(map(float, pose.p)),
                "rot": list(map(float, pose.q)),
                "articulation": True,
            }
        except Exception:
            continue
    return objs


def _locate_urdf(rt: str, name: str) -> str | None:
    """Find a URDF-object's ``mobility.urdf`` under ``assets/objects/<name>/`` (returns relpath)."""
    base = os.path.join(rt, "assets", "objects", name)
    direct = os.path.join(base, "mobility.urdf")
    if os.path.exists(direct):
        return os.path.relpath(direct, rt)
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            cand = os.path.join(base, sub, "mobility.urdf")
            if os.path.exists(cand):
                return os.path.relpath(cand, rt)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="beat_block_hammer", help="RoboTwin task name (envs/<task>.py)")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--max-seeds", type=int, default=25, help="seeds to try before giving up")
    ap.add_argument("--save-freq", type=int, default=10, help="record one frame every N sim steps")
    ap.add_argument("--out", required=True, help="output pickle path")
    args = ap.parse_args(argv)

    pt = _load_passthrough()
    work_dir = os.path.join(pt.robotwin_dir(), "data", "_rv_bridge")
    os.makedirs(work_dir, exist_ok=True)
    # Record the real mesh file RoboTwin loads per actor (for mesh-faithful replay).
    mesh_sink = _install_mesh_hook(pt)
    # State-only: skip RGB/point-cloud rendering so the seed search stays fast.
    # qpos -> command-target ``vector``; endpose -> achieved end-effector world
    # pose per arm. Together with the achieved ``real_vector`` (injected below)
    # these are the three signals the RoboVerse parity harness compares against.
    data_type = {k: False for k in ("rgb", "third_view", "depth", "pointcloud", "observer", "endpose", "qpos")}
    data_type["qpos"] = True
    data_type["endpose"] = True

    success = None
    for seed in range(args.max_seeds):
        try:
            env = pt._make_robotwin_env(
                task_name=args.task,
                task_config=args.task_config,
                seed=seed,
                save_data=True,
                save_path=work_dir,
                save_freq=args.save_freq,
                data_type=data_type,
                render_freq=0,
            )
        except Exception as e:
            print(f"[seed {seed}] setup failed: {type(e).__name__}: {e}")
            continue
        _patch_capture_real_state(env)
        init_objects = _capture_objects(env)
        try:
            env.play_once()
            ok = bool(env.plan_success) and bool(env.check_success())
        except Exception as e:
            print(f"[seed {seed}] play_once error: {type(e).__name__}: {e}")
            ok = False
        print(f"[seed {seed}] plan={getattr(env, 'plan_success', None)} success={ok} frames={env.FRAME_IDX}")
        if ok:
            cache = sorted(
                glob.glob(os.path.join(work_dir, ".cache", f"episode{env.ep_num}", "*.pkl")),
                key=lambda p: int(os.path.basename(p)[:-4]),
            )
            frames = [pickle.load(open(p, "rb")) for p in cache]
            vectors = [np.asarray(f["joint_action"]["vector"], dtype=float) for f in frames]
            # Achieved arm qpos (injected by _patch_capture_real_state) and achieved
            # end-effector world poses — the signals for honest, non-circular parity.
            real_vectors = [
                np.asarray(f["joint_action"]["real_vector"], dtype=float)
                for f in frames
                if "real_vector" in f.get("joint_action", {})
            ]
            left_endpose = [np.asarray(f["endpose"]["left_endpose"], dtype=float) for f in frames if f.get("endpose")]
            right_endpose = [np.asarray(f["endpose"]["right_endpose"], dtype=float) for f in frames if f.get("endpose")]
            # Per-frame object pose trajectory (manipulated actors only) + their meshes,
            # so the RoboVerse replay can load real meshes and we can measure object parity.
            obj_names = [n for n in init_objects if n not in _INFRA_ACTORS]
            object_traj = {
                n: np.asarray([
                    f["object_poses"][n] for f in frames if f.get("object_poses") and n in f["object_poses"]
                ])
                for n in obj_names
            }
            # Per-object asset: exact mesh from the create_actor hook, else a scanned
            # mobility.urdf for URDF-articulation objects (pot/cabinet/laptop/...).
            object_meshes = {}
            for n in obj_names:
                if n in mesh_sink:
                    object_meshes[n] = {**mesh_sink[n], "type": "mesh"}
                else:
                    urdf = _locate_urdf(pt.robotwin_dir(), n)
                    if urdf:
                        object_meshes[n] = {"urdf": urdf, "type": "urdf"}
            success = {
                "task": args.task,
                "seed": seed,
                "vectors": vectors,  # (T, 14) command target: [L_arm(6), L_grip, R_arm(6), R_grip]
                "real_vectors": real_vectors,  # (T, 14) achieved qpos, same layout (empty if capture failed)
                "left_endpose": left_endpose,  # (T, 7) achieved EE world pose [x,y,z, qw,qx,qy,qz]
                "right_endpose": right_endpose,  # (T, 7)
                "init_objects": init_objects,
                "object_traj": object_traj,  # {actor_name: (T, 7)} manipulated-object world-pose trajectory
                "object_meshes": object_meshes,  # {actor_name: {visual: relpath, scale, model_id}}
                "left_gripper_scale": list(env.robot.left_gripper_scale),
                "right_gripper_scale": list(env.robot.right_gripper_scale),
            }
            env.close_env()
            break
        env.close_env()

    if success is None:
        print(f"NO_SUCCESS: no successful episode in {args.max_seeds} seeds for {args.task!r}")
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(success, f)
    print(
        f"SAVED {len(success['vectors'])} frames (seed {success['seed']}) -> {args.out}; "
        f"achieved_qpos={len(success['real_vectors'])} endpose={len(success['left_endpose'])}; "
        f"obj_traj={ {k: len(v) for k, v in success['object_traj'].items()} } meshes={list(success['object_meshes'])}; "
        f"objects: {list(success['init_objects'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
