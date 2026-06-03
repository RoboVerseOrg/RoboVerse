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
    ap.add_argument(
        "--replay-bridge",
        default=None,
        help="drive the env from this bridge pkl (exact 1:1 episode) instead of play_once",
    )
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

    # CRITICAL for instance-randomized tasks: RoboTwin's setup_demo picks each object's mesh via
    # np.random.choice(model_ids), and that choice is NOT reproducible across collection vs this
    # re-render (a fresh setup_demo at the same seed yields *different* bases, e.g. base10/base14
    # instead of the collected base1/base6). Left alone, the native side would show different
    # bottle/object INSTANCES than the bridge -> the side-by-side compares two different episodes.
    # Force the SAME instances the bridge recorded by hooking get_glb_or_obj_file to return the
    # recorded visual mesh for the k-th create_actor of each model name (in creation order, which
    # matches how the bridge disambiguated object_meshes: name, name#1, name#2 ...).
    if args.replay_bridge:
        import importlib
        import pickle as _pk

        _br0 = _pk.load(open(args.replay_bridge, "rb"))
        _meshes = _br0.get("object_meshes", {})
        # Per category (model name = key without the #k suffix), ordered list of recorded visual
        # relpaths. e.g. {"001_bottle": ["assets/.../base1.glb", "assets/.../base6.glb"]}.
        _recorded: dict[str, list[str]] = {}
        for k, m in _meshes.items():
            if not isinstance(m, dict) or not m.get("visual"):
                continue
            cat = k.split("#")[0]
            _recorded.setdefault(cat, []).append(m["visual"])
        _used: dict[str, int] = {}
        _rt = pt.robotwin_dir()
        try:
            # `envs` only resolves once the RoboTwin checkout is on sys.path AND it's the CWD
            # (importing envs.utils reads a relative assets/objects/objaverse/list.json). Both are
            # normally done inside _make_robotwin_env; do them now so we can patch the mesh resolver
            # BEFORE setup_demo. chdir is what _make_robotwin_env does anyway, so it's idempotent.
            if _rt not in sys.path:
                sys.path.insert(0, _rt)
            os.chdir(_rt)
            cau = importlib.import_module("envs.utils.create_actor")
            _orig_glb = cau.get_glb_or_obj_file

            def _forced_glb(modeldir, model_id):
                cat = os.path.basename(str(modeldir).rstrip("/"))
                if cat in ("visual", "collision"):
                    cat = os.path.basename(os.path.dirname(str(modeldir).rstrip("/")))
                want = _recorded.get(cat)
                if want:
                    i = _used.get(cat, 0)
                    if i < len(want):
                        rel = want[i]
                        # create_actor calls this twice per actor: collision dir first, then
                        # visual. Map BOTH to the same recorded instance; advance only on the
                        # visual call so the next actor gets the next recorded mesh.
                        is_collision = f"{os.sep}collision{os.sep}" in str(modeldir)
                        if is_collision:
                            rel = rel.replace(f"{os.sep}visual{os.sep}", f"{os.sep}collision{os.sep}")
                        p = os.path.join(_rt, rel)
                        if os.path.exists(p):
                            if not is_collision:
                                _used[cat] = i + 1  # advance only once the actor is committed
                            # Return the SAME type get_glb_or_obj_file does (a pathlib.Path; callers
                            # call .exists() on it), not a bare str.
                            from pathlib import Path as _Path

                            return _Path(p)
                return _orig_glb(modeldir, model_id)

            cau.get_glb_or_obj_file = _forced_glb
            eu = sys.modules.get("envs.utils")
            if eu is not None and hasattr(eu, "get_glb_or_obj_file"):
                eu.get_glb_or_obj_file = _forced_glb
        except Exception as e:
            print(f"[native_render] could not force recorded instances ({type(e).__name__}: {e})")

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

    def render_frame():
        env.scene.update_render()
        cam.take_picture()
        rgb = cam.get_picture("Color")
        frames.append((np.clip(np.asarray(rgb)[..., :3], 0, 1) * 255).astype(np.uint8))

    if args.replay_bridge:
        # Drive the native env from the SAME bridge trajectory the RoboVerse replay
        # uses (teleport robot arm qpos + object poses/joints per frame), instead of
        # re-planning via play_once -> the two videos are the *identical* episode,
        # frame-for-frame, so any remaining difference is purely the render engine.
        import pickle

        br = pickle.load(open(args.replay_bridge, "rb"))
        real = br.get("real_vectors") or br["vectors"]
        otraj = br.get("object_traj", {})
        ojoint = br.get("object_joint_traj", {})
        rob = env.robot.left_entity
        act = [j.get_name() for j in rob.get_active_joints()]
        lidx = [act.index(n) for n in env.robot.left_arm_joints_name]
        ridx = [act.index(n) for n in env.robot.right_arm_joints_name]
        # Gripper joints: (qpos-index, mimic mult, offset); value = scale[0] + norm*(scale[1]-scale[0]).
        lgrip = [(act.index(j[0].get_name()), j[1], j[2]) for j in env.robot.left_gripper]
        rgrip = [(act.index(j[0].get_name()), j[1], j[2]) for j in env.robot.right_gripper]
        lgs, rgs = env.robot.left_gripper_scale, env.robot.right_gripper_scale

        # Disambiguate duplicate-named actors EXACTLY as the bridge did when it recorded
        # object_traj (name, name#1, name#2, ... in get_all_actors creation order). A plain
        # {name: actor} dict collapses duplicates (e.g. put_bottles_dustbin's three 114_bottle,
        # place_cans_plasticbox's multiple cans) to a single entry -> only ONE of them gets its
        # recorded pose and the rest stay at their setup_demo positions = the native side looks
        # misaligned / wrong-instance vs the replay. Counting by occurrence makes the keys match
        # object_traj's disambiguated keys so EVERY duplicate is positioned.
        def _disambiguated(seq):
            out, counts = {}, {}
            for a in seq:
                nm = a.get_name()
                n = counts.get(nm, 0)
                counts[nm] = n + 1
                out[nm if n == 0 else f"{nm}#{n}"] = a
            return out

        actors = _disambiguated(env.scene.get_all_actors())
        arts = _disambiguated([a for a in env.scene.get_all_articulations() if a.get_name() != rob.get_name()])
        for t in range(len(real)):
            q = np.asarray(rob.get_qpos(), dtype=float).copy()
            for i, ix in enumerate(lidx):
                q[ix] = real[t][i]
            for i, ix in enumerate(ridx):
                q[ix] = real[t][7 + i]
            lbase = lgs[0] + float(real[t][6]) * (lgs[1] - lgs[0])
            for ix, mult, off in lgrip:
                q[ix] = lbase * mult + off
            rbase = rgs[0] + float(real[t][13]) * (rgs[1] - rgs[0])
            for ix, mult, off in rgrip:
                q[ix] = rbase * mult + off
            rob.set_qpos(q)
            for name, tr in otraj.items():
                p = tr[min(t, len(tr) - 1)]
                pose = sapien.Pose(p[:3], p[3:7])
                if name in actors:
                    actors[name].set_pose(pose)
                elif name in arts:
                    arts[name].set_root_pose(pose)
                    if name in ojoint:
                        jt = ojoint[name]
                        arts[name].set_qpos(np.asarray(jt[min(t, len(jt) - 1)], dtype=float))
            render_frame()
        ok = True
        print(f"[{args.task}] bridge-replay rendered {len(frames)} frames (exact trajectory)")
    else:
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
