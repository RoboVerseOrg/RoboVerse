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

    def get_obs_with_real():
        d = orig_get_obs()
        try:
            real = list(robot.get_left_arm_real_jointState()) + list(robot.get_right_arm_real_jointState())
            d.setdefault("joint_action", {})["real_vector"] = np.asarray(real, dtype=float)
        except Exception:
            pass  # achieved-state capture is best-effort; target vector is always present
        return d

    env.get_obs = get_obs_with_real


def _capture_objects(env) -> dict:
    """Record every scene actor's initial world pose (wxyz quaternion)."""
    objs: dict = {}
    try:
        actors = env.scene.get_all_actors()
    except Exception:
        return objs
    for actor in actors:
        try:
            pose = actor.get_pose()
            objs[actor.get_name()] = {"pos": list(map(float, pose.p)), "rot": list(map(float, pose.q))}
        except Exception:
            continue
    return objs


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
            success = {
                "task": args.task,
                "seed": seed,
                "vectors": vectors,  # (T, 14) command target: [L_arm(6), L_grip, R_arm(6), R_grip]
                "real_vectors": real_vectors,  # (T, 14) achieved qpos, same layout (empty if capture failed)
                "left_endpose": left_endpose,  # (T, 7) achieved EE world pose [x,y,z, qw,qx,qy,qz]
                "right_endpose": right_endpose,  # (T, 7)
                "init_objects": init_objects,
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
        f"objects: {list(success['init_objects'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
