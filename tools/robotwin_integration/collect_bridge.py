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
            qpos = {}
            jnames = {}
            counts: dict = {}  # disambiguate duplicate actor names (e.g. two 001_bottle)
            for actor in scene.get_all_actors():
                p = actor.get_pose()
                key = _disambiguate(actor.get_name(), counts)
                poses[key] = np.asarray([*p.p, *p.q], dtype=float)  # [x,y,z, qw,qx,qy,qz]
            for art in scene.get_all_articulations():
                name = art.get_name()
                if name in robot_arts:
                    continue
                key = _disambiguate(name, counts)
                p = art.get_root_pose() if hasattr(art, "get_root_pose") else art.get_pose()
                poses[key] = np.asarray([*p.p, *p.q], dtype=float)
                name = key  # keep qpos/jnames keyed consistently with poses
                # Articulation joint qpos (+ names for ordering) -> replay opening
                # doors/lids/drawers 1:1.
                try:
                    q = art.get_qpos()
                    if q is not None and len(q):
                        qpos[name] = np.asarray(q, dtype=float)
                        jnames[name] = [j.get_name() for j in art.get_active_joints()]
                except Exception:
                    pass
            d["object_poses"] = poses
            d["object_qpos"] = qpos
            d["object_joint_names"] = jnames
        except Exception:
            pass
        return d

    env.get_obs = get_obs_with_real


# Infrastructure actors (not manipulated targets) we never need a mesh for.
_INFRA_ACTORS = {"ground", "wall", "table", "table_wall"}


def _disambiguate(name: str, counts: dict) -> str:
    """Unique key for ``name``, suffixing duplicates as ``name#1``, ``name#2``, ...

    RoboTwin tasks may create several actors with the SAME name (e.g.
    pick_dual_bottles makes two ``001_bottle``); keying captured poses/meshes by raw
    name silently drops all but one ("missing object"). Counting by occurrence order
    keeps every instance, and because SAPIEN preserves actor creation order, the k-th
    same-named actor in ``get_all_actors()`` lines up with the k-th create_actor call
    the mesh hook saw -> pose and mesh disambiguation agree.
    """
    n = counts.get(name, 0)
    counts[name] = n + 1
    return name if n == 0 else f"{name}#{n}"


def resolve_urdf_instance_dir(objects_root: str, modelname: str, modelid):
    """Resolve the EXACT URDF instance dir RoboTwin's create_sapien_urdf_obj uses.

    This must match RoboTwin 1:1, because modelid (random per episode) selects which
    physical object is loaded. RoboTwin builds the candidate list as the subdirs of
    ``<objects_root>/<modelname>`` that are directories, EXCLUDING ``visual`` (a
    shared mesh dir, not an instance), sorted by their numeric name, then indexes by
    modelid (falling back to a dir whose name equals modelid). Including ``visual``
    (it sorts to index 0) shifts every instance -> "same category, wrong object".
    We additionally require ``mobility.urdf`` to be present, which is what makes a
    subdir a real URDF instance. Returns ``<objects_root>/<modelname>`` if modelid is
    None or unresolved.
    """
    base = os.path.join(objects_root, modelname)
    if modelid is None:
        return base
    subs = sorted(
        (
            d
            for d in os.listdir(base)
            if d != "visual"
            and os.path.isdir(os.path.join(base, d))
            and os.path.exists(os.path.join(base, d, "mobility.urdf"))
        ),
        key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0),
    )
    if modelid < len(subs):
        return os.path.join(base, subs[modelid])
    for d in subs:  # fall back to matching the dir name to the id
        if d.isdigit() and int(d) == modelid:
            return os.path.join(base, d)
    return base


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
    # Per-instance capture so duplicate-named objects (two 001_bottle, three blocks,
    # two 002_bowl) are each kept under disambiguated keys (name, name#1, ...) that
    # line up with the disambiguated per-frame poses. ``_pending`` buffers the mesh a
    # create_actor resolves (collision then visual fire before the actor commits);
    # ``counts`` is the shared occurrence counter matching the pose-capture order.
    _pending: dict = {}
    counts: dict = {}
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
            # Buffer this actor's mesh in _pending; the create_actor/rand_create_actor
            # wrapper commits it to a disambiguated sink key when the actor is built.
            if name not in _pending or is_visual:
                _pending[name] = {"visual": rel, "scale": scale, "model_id": model_id, "type": "mesh"}
        except Exception:
            pass
        return f

    cau.get_glb_or_obj_file = hooked

    def _commit_actor(modelname, is_static):
        # Move the buffered mesh for this actor to a unique sink key, in creation
        # order, so the k-th same-named actor lines up with the k-th captured mesh.
        mesh = _pending.pop(modelname, None)
        if mesh is None:
            return
        mesh["is_static"] = bool(is_static)
        sink[_disambiguate(modelname, counts)] = mesh

    # Also capture create_box primitives (target markers / blocks) so the replay
    # uses their real half-size + color instead of a generic red proxy cube.
    if hasattr(cau, "create_box"):
        orig_box = cau.create_box

        def hooked_box(scene, pose, half_size, color=None, is_static=False, name="", *a, **k):
            try:
                if name:
                    hs = [float(x) for x in half_size] if hasattr(half_size, "__iter__") else [float(half_size)] * 3
                    sink[_disambiguate(name, counts)] = {
                        "type": "box",
                        "half_size": hs,
                        "color": [float(c) for c in color] if color is not None else None,
                        "is_static": bool(is_static),
                    }
            except Exception:
                pass
            return orig_box(scene, pose, half_size, *a, color=color, is_static=is_static, name=name, **k)

        cau.create_box = hooked_box
        # Tasks do ``from .utils import *``, which re-exports create_box into the
        # ``envs.utils`` package namespace -- a SEPARATE binding the task copies at
        # import time. Patch it there too (before the task is imported) so the hook
        # actually sees the call.
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, "create_box"):
            eu.create_box = hooked_box

    # Capture the OTHER primitive creators too -- create_sphere (dump_bin garbage),
    # create_cylinder, create_visual_box (stamp_seal). Without these the objects have
    # a recorded pose but no shape, so the replay drops them = "missing objects".
    def _patch(fnname, record):
        if not hasattr(cau, fnname):
            return
        orig = getattr(cau, fnname)

        def wrapped(scene, pose, *a, **k):
            try:
                name = k.get("name", "")
                if name:
                    sink[_disambiguate(name, counts)] = record(a, k)
            except Exception:
                pass
            return orig(scene, pose, *a, **k)

        setattr(cau, fnname, wrapped)
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, fnname):
            setattr(eu, fnname, wrapped)

    def _sphere_rec(a, k):
        radius = k.get("radius", a[0] if a else 0.02)
        color = k.get("color", a[1] if len(a) >= 2 else None)
        return {
            "type": "sphere",
            "radius": float(radius),
            "color": [float(c) for c in color] if color is not None else None,
            "is_static": bool(k.get("is_static", False)),
        }

    def _cylinder_rec(a, k):
        radius = k.get("radius", a[0] if a else 0.02)
        half_length = k.get("half_length", a[1] if len(a) >= 2 else 0.02)
        color = k.get("color", a[2] if len(a) >= 3 else None)
        return {
            "type": "cylinder",
            "radius": float(radius),
            "height": float(half_length) * 2.0,
            "color": [float(c) for c in color] if color is not None else None,
        }

    def _vbox_rec(a, k):
        hs = k.get("half_size", a[0] if a else [0.02, 0.02, 0.02])
        color = k.get("color", a[1] if len(a) >= 2 else None)
        hs = [float(x) for x in hs] if hasattr(hs, "__iter__") else [float(hs)] * 3
        return {
            "type": "box",
            "half_size": hs,
            "color": [float(c) for c in color] if color is not None else None,
            "is_static": True,
        }

    _patch("create_sphere", _sphere_rec)
    _patch("create_cylinder", _cylinder_rec)
    _patch("create_visual_box", _vbox_rec)

    # Capture create_actor's is_static flag (e.g. baskets/scales loaded static) so
    # the replay can match RoboTwin's static/dynamic choice exactly rather than
    # inferring it from whether the object moved. create_actor signature:
    # (scene, pose, modelname, scale, convex, is_static, model_id).
    if hasattr(cau, "create_actor"):
        orig_ca = cau.create_actor

        def hooked_ca(scene, pose, modelname, *a, **k):
            r = orig_ca(scene, pose, modelname, *a, **k)
            try:
                is_static = k.get("is_static", a[2] if len(a) >= 3 else False)
                _commit_actor(modelname, is_static)
            except Exception:
                pass
            return r

        cau.create_actor = hooked_ca
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, "create_actor"):
            eu.create_actor = hooked_ca

    # rand_create_actor(modelname, ...) calls create_actor via its OWN binding (so the
    # cau.create_actor hook above is bypassed for it) -- wrap it so the per-instance
    # commit still fires for the many tasks that randomly place mesh objects.
    try:
        rca = importlib.import_module("envs.utils.rand_create_actor")
    except Exception:
        rca = None
    if rca is not None and hasattr(rca, "rand_create_actor"):
        orig_rca = rca.rand_create_actor

        def hooked_rca(scene, modelname, *a, **k):
            r = orig_rca(scene, modelname, *a, **k)
            try:
                _commit_actor(modelname, k.get("is_static", False))
            except Exception:
                pass
            return r

        rca.rand_create_actor = hooked_rca
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, "rand_create_actor"):
            eu.rand_create_actor = hooked_rca

    # Capture URDF/articulation objects (pot/cabinet/laptop/microwave/...) at their
    # EXACT instance + scale. These load via create_sapien_urdf_obj, NOT through the
    # mesh resolver above, so without this the replay (a) scanned for *any*
    # mobility.urdf -> wrong instance ("same category, different object"), and (b)
    # had no scale -> the model rendered the wrong size. The articulation is named
    # ``modelname``; modelid selects the instance subdir (sorted by number) and the
    # scale lives in that subdir's model_data.json -- replicate RoboTwin's resolution.
    def _resolve_modeldir(modelname, modelid):
        # Tasks run with cwd = the RoboTwin checkout, so assets live at ./assets/objects.
        return resolve_urdf_instance_dir(_os.path.join("assets", "objects"), modelname, modelid)

    def _record_urdf(modelname, modelid, scale):
        try:
            modeldir = _resolve_modeldir(modelname, modelid)
            eff_scale = scale
            jf = _os.path.join(modeldir, "model_data.json")
            if _os.path.exists(jf):
                eff_scale = _json.load(open(jf)).get("scale", scale)
            sink[_disambiguate(modelname, counts)] = {
                "urdf": _os.path.join(modeldir, "mobility.urdf"),
                "scale": eff_scale,
                "modelid": modelid,
                "type": "urdf",
            }
        except Exception:
            pass

    # Tasks call ``rand_create_sapien_urdf_obj(modelname, modelid, ...)`` (which
    # internally calls create_sapien_urdf_obj via rand_create_actor's OWN binding,
    # so patching create_actor alone is bypassed). Wrap the rand entry point in its
    # module + the envs.utils re-export (rca imported above for rand_create_actor).
    if rca is not None and hasattr(rca, "rand_create_sapien_urdf_obj"):
        orig_rurdf = rca.rand_create_sapien_urdf_obj

        def hooked_rurdf(scene, modelname, modelid, *a, **k):
            r = orig_rurdf(scene, modelname, modelid, *a, **k)
            _record_urdf(modelname, modelid, k.get("scale", 1.0))
            return r

        rca.rand_create_sapien_urdf_obj = hooked_rurdf
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, "rand_create_sapien_urdf_obj"):
            eu.rand_create_sapien_urdf_obj = hooked_rurdf
    # Also wrap the direct (non-rand) entry point for tasks that use it.
    if hasattr(cau, "create_sapien_urdf_obj"):
        orig_urdf = cau.create_sapien_urdf_obj

        def hooked_urdf(scene, pose, modelname, scale=1.0, modelid=None, *a, **k):
            r = orig_urdf(scene, pose, modelname, scale, modelid, *a, **k)
            _record_urdf(modelname, modelid, scale)
            return r

        cau.create_sapien_urdf_obj = hooked_urdf
        eu = _sys.modules.get("envs.utils")
        if eu is not None and hasattr(eu, "create_sapien_urdf_obj"):
            eu.create_sapien_urdf_obj = hooked_urdf
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
    counts: dict = {}  # disambiguate duplicate names, matching the per-frame capture
    for actor in actors:
        try:
            pose = actor.get_pose()
            key = _disambiguate(actor.get_name(), counts)
            objs[key] = {"pos": list(map(float, pose.p)), "rot": list(map(float, pose.q))}
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
            objs[_disambiguate(art.get_name(), counts)] = {
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
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="also capture camera RGB per frame (for IL policy training; larger pickle, slower)",
    )
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="with --rgb, only capture these cameras (e.g. head_camera) -> smaller/faster; default all",
    )
    ap.add_argument("--out", required=True, help="output pickle path (single demo) or directory (--num-demos>1)")
    ap.add_argument(
        "--num-demos",
        type=int,
        default=1,
        help="collect this many distinct successful episodes (>1 saves demo_XXXX.pkl under --out dir)",
    )
    ap.add_argument("--start-seed", type=int, default=0, help="first seed to try (vary to get distinct demos)")
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
    data_type["rgb"] = bool(args.rgb)  # IL training needs per-frame camera RGB

    collected = 0  # number of successful episodes saved so far
    saved_paths: list[str] = []
    for seed in range(args.start_seed, args.start_seed + args.max_seeds):
        if collected >= args.num_demos:
            break
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
        if args.rgb:
            # RoboTwin's setup_demo enables the RT shader + OIDN denoiser; OIDN can
            # deadlock headless on a long episode ("OIDN Error: invalid handle"),
            # hanging multi-demo RGB collection. Drop the denoiser (RT geometry still
            # renders) -- same workaround as native_render.py.
            try:
                import sapien

                sapien.render.set_ray_tracing_denoiser("none")
            except Exception:
                pass
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
            # Per-frame joint qpos (+ joint names) for articulated objects (doors/lids/drawers).
            object_joint_traj = {}
            object_joint_names = {}
            for n in obj_names:
                seq = [f["object_qpos"][n] for f in frames if f.get("object_qpos") and n in f.get("object_qpos", {})]
                if seq:
                    object_joint_traj[n] = np.asarray(seq)
                    for f in frames:
                        if n in f.get("object_joint_names", {}):
                            object_joint_names[n] = f["object_joint_names"][n]
                            break
            # Per-object asset: exact mesh from the create_actor hook, else a scanned
            # mobility.urdf for URDF-articulation objects (pot/cabinet/laptop/...).
            object_meshes = {}
            for n in obj_names:
                if n in mesh_sink:
                    object_meshes[n] = {**mesh_sink[n], "type": mesh_sink[n].get("type", "mesh")}
                else:
                    urdf = _locate_urdf(pt.robotwin_dir(), n)
                    if urdf:
                        object_meshes[n] = {"urdf": urdf, "type": "urdf"}
            # Per-frame camera RGB for IL policy training (only when --rgb). RoboTwin's
            # get_obs nests frames under f["observation"][cam]["rgb"] (uint8 HxWx3);
            # capture every camera so DP/ACT can train on head + wrist views.
            rgb_traj = {}
            if args.rgb:
                cam_names = set()
                for f in frames:
                    cam_names.update((f.get("observation") or {}).keys())
                if args.cameras:  # restrict to requested cameras (smaller pickle)
                    cam_names &= set(args.cameras)
                for cam in sorted(cam_names):
                    seq = [
                        np.asarray(f["observation"][cam]["rgb"], dtype=np.uint8)
                        for f in frames
                        if f.get("observation", {}).get(cam, {}).get("rgb") is not None
                    ]
                    if seq:
                        rgb_traj[cam] = np.asarray(seq)
            success = {
                "task": args.task,
                "seed": seed,
                "vectors": vectors,  # (T, 14) command target: [L_arm(6), L_grip, R_arm(6), R_grip]
                "real_vectors": real_vectors,  # (T, 14) achieved qpos, same layout (empty if capture failed)
                "left_endpose": left_endpose,  # (T, 7) achieved EE world pose [x,y,z, qw,qx,qy,qz]
                "right_endpose": right_endpose,  # (T, 7)
                "init_objects": init_objects,
                "object_traj": object_traj,  # {actor_name: (T, 7)} manipulated-object world-pose trajectory
                "object_joint_traj": object_joint_traj,  # {actor_name: (T, ndof)} articulated-object joint qpos
                "object_joint_names": object_joint_names,  # {actor_name: [joint names]} order of the qpos cols
                "object_meshes": object_meshes,  # {actor_name: {visual: relpath, scale, model_id}}
                "left_gripper_scale": list(env.robot.left_gripper_scale),
                "right_gripper_scale": list(env.robot.right_gripper_scale),
                "rgb": rgb_traj,  # {camera_name: (T, H, W, 3) uint8} -- only when --rgb (else {})
            }
            env.close_env()
            # Single-demo (default): write to --out verbatim (backward compatible).
            # Multi-demo: --out is a directory of demo_XXXX.pkl files.
            if args.num_demos == 1:
                out_path = args.out
            else:
                out_path = os.path.join(args.out, f"demo_{collected:04d}.pkl")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "wb") as fp:
                pickle.dump(success, fp)
            saved_paths.append(out_path)
            collected += 1
            print(
                f"  [demo {collected}/{args.num_demos}] seed {seed}: {len(success['vectors'])} frames, "
                f"rgb={ {k: tuple(v.shape) for k, v in success['rgb'].items()} } -> {out_path}"
            )
            continue
        env.close_env()

    if collected == 0:
        print(f"NO_SUCCESS: no successful episode in {args.max_seeds} seeds for {args.task!r}")
        return 2
    if args.num_demos > 1:
        print(f"SAVED {collected} demo(s) -> {args.out}")
        return 0

    # Single-demo summary (the loop already wrote args.out; print the detailed log).
    print(
        f"SAVED {len(success['vectors'])} frames (seed {success['seed']}) -> {args.out}; "
        f"achieved_qpos={len(success['real_vectors'])} endpose={len(success['left_endpose'])}; "
        f"obj_traj={ {k: len(v) for k, v in success['object_traj'].items()} } meshes={list(success['object_meshes'])}; "
        f"rgb={ {k: tuple(v.shape) for k, v in success['rgb'].items()} }; "
        f"objects: {list(success['init_objects'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
