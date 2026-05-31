"""Render a native RoboTwin episode from the observer camera (ground truth).

Runs in the ``robotwin`` conda env. Drives a task through the passthrough factory
at a given seed and captures the built-in ``observer_camera`` RGB every saved
frame, writing an mp4. This is the *ground-truth* half of the native-vs-RoboVerse
side-by-side: the RoboVerse replay (``mesh_replay_robotwin.py``) renders the same
scene from the SAME observer pose, so the two videos can be composited frame-for-
frame to verify true 1:1 object + robot alignment.

The observer pose is read straight from RoboTwin (``envs/camera/camera.py``):
pos [0, 0.23, 1.33], forward [0,-1,-1.02], left [1,0,0].

Run::

    conda run -n robotwin env MUJOCO_GL=egl python \\
        tools/robotwin_integration/native_render.py --task move_can_pot --seed 0 \\
        --out outputs/robotwin_coverage/native_move_can_pot.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PASSTHROUGH = os.path.join(_HERE, "..", "..", "roboverse_pack", "tasks", "robotwin", "_passthrough.py")


def _load_passthrough():
    spec = importlib.util.spec_from_file_location("rt_passthrough", _PASSTHROUGH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="move_can_pot")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=15)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.0, 0.6, 1.35])
    ap.add_argument("--cam-lookat", type=float, nargs=3, default=[0.0, -0.3, 0.78])
    ap.add_argument("--fovy", type=float, default=55.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import imageio.v2 as imageio
    import sapien

    pt = _load_passthrough()
    work_dir = os.path.join(pt.robotwin_dir(), "data", "_rv_bridge")
    os.makedirs(work_dir, exist_ok=True)
    # All data_type off: get_obs still fires (it always _update_render()s the scene),
    # and we capture our OWN match camera below. Rendering RoboTwin's observer too
    # (third_view) just doubles the per-frame RT cost on long episodes.
    data_type = {k: False for k in ("rgb", "third_view", "depth", "pointcloud", "observer", "endpose", "qpos")}

    env = pt._make_robotwin_env(
        task_name=args.task, task_config=args.task_config, seed=args.seed,
        save_data=False, data_type=data_type, render_freq=0,
    )  # fmt: skip

    # RoboTwin's setup_demo enables the RT shader + the OIDN denoiser; OIDN can
    # deadlock/leak over a long episode in headless ("OIDN Error: invalid handle").
    # Drop the denoiser (RT geometry/shadows still render) so long tasks don't hang.
    try:
        sapien.render.set_ray_tracing_denoiser("none")
    except Exception:
        pass

    # Add a roll=0 (world-up) camera matching RoboVerse's PinholeCameraCfg convention
    # (sapien3.py: yaw/pitch from look-dir, roll hardcoded 0), so native + RoboVerse
    # render from the *identical* viewpoint and composite frame-for-frame.
    pos = np.array(args.cam_pos, dtype=float)
    look = np.array(args.cam_lookat, dtype=float)
    fwd = look - pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left /= np.linalg.norm(left) + 1e-9
    up = np.cross(fwd, left)
    mat = np.eye(4)
    mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = fwd, left, up, pos
    cam = env.scene.add_camera(
        name="match", width=args.width, height=args.height, fovy=np.deg2rad(args.fovy), near=0.05, far=100
    )
    cam.entity.set_pose(sapien.Pose(mat))

    frames: list = []
    orig_get_obs = env.get_obs

    def hooked():
        d = orig_get_obs()
        try:
            env.scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")  # HxWx4 float [0,1]
            frames.append((np.clip(np.asarray(rgb)[..., :3], 0, 1) * 255).astype(np.uint8))
        except Exception as e:
            print(f"camera grab failed: {type(e).__name__}: {e}")
        return d

    env.get_obs = hooked
    env.save_freq = args.save_freq
    env.save_data = True  # so _take_picture (which calls get_obs) fires every save_freq steps

    try:
        env.play_once()
        ok = bool(getattr(env, "plan_success", False)) and bool(env.check_success())
    except Exception as e:
        print(f"play_once error: {type(e).__name__}: {e}")
        ok = False
    print(f"[{args.task} seed {args.seed}] success={ok} observer_frames={len(frames)}")
    env.close_env()

    if not frames:
        print("NO_FRAMES")
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=20)
    print(f"SAVED {len(frames)} observer frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
