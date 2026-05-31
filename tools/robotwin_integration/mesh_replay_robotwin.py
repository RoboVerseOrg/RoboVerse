"""Mesh-faithful RoboTwin replay in RoboVerse + object-pose parity.

Two modes, both in the ``roboverse`` env, both loading the REAL RoboTwin object
meshes (not a primitive proxy):

* ``--mode kinematic`` (default): full state replay -- every frame the robot is
  set to RoboTwin's achieved qpos and each manipulated object to its recorded
  world pose. This is faithful *playback*: the rendered scene matches RoboTwin
  frame-for-frame (real meshes in the right place). Use it for the headline
  side-by-side video. It exercises no physics; it is playback, not reproduction.

* ``--mode physics``: the robot is driven by RoboTwin's command-target stream
  (dof-position targets, same as the joint-parity harness) and the manipulated
  objects are DYNAMIC -- they move only by contact. We then compare the
  RoboVerse object trajectory against RoboTwin's recorded one and report the
  object-pose delta (position L2 + quaternion geodesic angle). This is the
  honest *task-level* 1:1 test: does the same robot motion push the object to
  the same place? It is contact-sensitive, so deltas are reported, not asserted.

Needs a bridge pickle collected by the (enhanced) ``collect_bridge.py`` carrying
``object_traj`` + ``object_meshes``.

Run::

    MUJOCO_GL=egl python tools/robotwin_integration/mesh_replay_robotwin.py \\
        --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl --mode kinematic --video
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import glob
import json
import os
import pickle

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from roboverse_pack.robots.aloha_agilex_cfg import AlohaAgilexCfg
from roboverse_pack.tasks.robotwin._convert import ROBOT_NAME, ROBOT_POS, ROBOT_ROT, bridge_to_v2, vector_to_dof

_INFRA = {"ground", "wall", "table", "table_wall"}

_URDF_TMPL = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual><origin xyz="0 0 0"/><geometry><mesh filename="{visual}" scale="{sx} {sy} {sz}"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><mesh filename="{collision}" scale="{sx} {sy} {sz}"/></geometry></collision>
    <inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>
  </link>
</robot>
"""


def _safe(name: str) -> str:
    """RoboVerse object name from a RoboTwin actor name (no leading digit)."""
    return "obj_" + name.replace("-", "_")


def _glb_to_urdf(visual_abs: str, scale, out_dir: str, name: str) -> str:
    """Wrap a mesh in a minimal single-link URDF (sapien loads GLB/OBJ via assimp).

    The collision geometry points at RoboTwin's dedicated ``collision/`` mesh when
    present (it is built for physics) rather than the dense visual mesh -- the
    sapien3 loader turns it into a convex hull, and a tight collision mesh avoids
    the init interpenetration that ejects concave dynamic objects.
    """
    os.makedirs(out_dir, exist_ok=True)
    sx, sy, sz = (float(s) for s in (scale if scale else (1, 1, 1)))
    collision_abs = visual_abs.replace(f"{os.sep}visual{os.sep}", f"{os.sep}collision{os.sep}")
    if not os.path.exists(collision_abs):
        collision_abs = visual_abs
    urdf = os.path.join(out_dir, f"{name}.urdf")
    with open(urdf, "w") as f:
        f.write(_URDF_TMPL.format(name=name, visual=visual_abs, collision=collision_abs, sx=sx, sy=sy, sz=sz))
    return urdf


def _urdf_to_glb(urdf_abs: str, out_dir: str, name: str) -> str | None:
    """Bake a URDF-articulation object (at rest config) into one textured GLB.

    sapien3's articulation loader renders the mobility.urdf geometry but drops its
    ``.mtl`` textures (-> gray). yourdfpy loads the URDF at its default joint
    config and trimesh exports the assembled scene to a GLB that *embeds* the
    textures, which then load like any rigid mesh. Static targets only (rest pose).
    """
    try:
        import trimesh  # noqa: F401
        import yourdfpy

        os.makedirs(out_dir, exist_ok=True)
        glb = os.path.join(out_dir, f"{name}.glb")
        if not os.path.exists(glb):
            yourdfpy.URDF.load(urdf_abs).scene.export(glb)
        return glb
    except Exception as e:
        log.warning(f"urdf->glb failed for {name} ({type(e).__name__}: {e}); falling back to articulation")
        return None


def _quat_angle(q1, q2) -> float:
    """Geodesic angle (rad) between two wxyz quaternions."""
    d = abs(float(np.dot(q1 / (np.linalg.norm(q1) + 1e-9), q2 / (np.linalg.norm(q2) + 1e-9))))
    return float(2.0 * np.arccos(min(1.0, d)))


def _replay_one(bridge: dict, args) -> dict:
    """Replay one bridge trajectory; render video and/or measure object parity."""
    task = bridge.get("task", "?")
    ls, rs = bridge["left_gripper_scale"], bridge["right_gripper_scale"]
    vectors = bridge["vectors"]
    real = bridge.get("real_vectors") or vectors
    object_traj = bridge.get("object_traj", {})
    object_meshes = bridge.get("object_meshes", {})
    object_joint_traj = bridge.get("object_joint_traj", {})
    object_joint_names = bridge.get("object_joint_names", {})
    # An articulated object whose joints actually move (door/lid/drawer opening)
    # must be replayed as an articulation; a static one can use the textured GLB bake.
    moving_artic = {n: jt for n, jt in object_joint_traj.items() if len(jt) and float(np.ptp(jt, axis=0).max()) > 0.02}
    manip = [n for n in object_traj if n not in _INFRA and len(object_traj[n])]
    log.info(f"[{task}] {len(vectors)} frames; manipulated objects: {manip}; meshes: {list(object_meshes)}")

    robot = AlohaAgilexCfg()
    if not robot.urdf_path:
        raise FileNotFoundError("ALOHA-AgileX URDF not found; extract RoboTwin embodiments.zip")

    # Build object cfgs: real mesh where we have one, else a small primitive proxy.
    kinematic = args.mode == "kinematic"
    phys = PhysicStateType.XFORM if kinematic else PhysicStateType.GEOM
    objects, name_map, artic_names = [], {}, set()
    for n in manip:
        rv = _safe(n)
        name_map[n] = rv
        mesh = object_meshes.get(n)
        if mesh and mesh.get("type") == "urdf":
            # URDF-articulation object (pot / cabinet / laptop). RoboTwin sets the
            # loader scale to model_data["scale"][0]; reproduce it or the object
            # loads at raw (huge) units.
            urdf_abs = os.path.join(args.robotwin_dir, mesh["urdf"])
            uscale = mesh.get("scale")
            if uscale is None:
                md = os.path.join(os.path.dirname(urdf_abs), "model_data.json")
                if os.path.exists(md):
                    s = json.load(open(md)).get("scale")
                    uscale = s[0] if isinstance(s, (list, tuple)) else s
            uscale = float(uscale or 1.0)
            if n in moving_artic:
                # Joints move (door/lid/drawer opens) -> must be an articulation so we
                # can drive its dof_pos per frame (the GLB bake is frozen at rest pose).
                artic_names.add(rv)
                objects.append(ArticulationObjCfg(name=rv, urdf_path=urdf_abs, fix_base_link=kinematic, scale=uscale))
                continue
            # Static URDF object: prefer a textured GLB baked from the URDF (rest pose)
            # so .mtl textures render; fall back to ArticulationObjCfg if the bake fails.
            glb = _urdf_to_glb(urdf_abs, "outputs/robotwin_coverage/_obj_glb", rv)
            if glb is not None:
                urdf = _glb_to_urdf(
                    os.path.abspath(glb), (uscale, uscale, uscale), "outputs/robotwin_coverage/_obj_urdf", rv
                )
                objects.append(RigidObjCfg(name=rv, urdf_path=urdf, physics=phys, fix_base_link=kinematic))
            else:
                artic_names.add(rv)
                objects.append(ArticulationObjCfg(name=rv, urdf_path=urdf_abs, fix_base_link=kinematic, scale=uscale))
        elif mesh and mesh.get("visual"):
            # Rigid mesh object: wrap the GLB/OBJ in a minimal URDF for sapien3.
            mesh_abs = os.path.join(args.robotwin_dir, mesh["visual"])
            urdf = _glb_to_urdf(mesh_abs, mesh.get("scale"), "outputs/robotwin_coverage/_obj_urdf", rv)
            objects.append(RigidObjCfg(name=rv, urdf_path=urdf, physics=phys, fix_base_link=kinematic))
        else:
            objects.append(PrimitiveCubeCfg(name=rv, size=(0.05, 0.05, 0.05), color=[0.8, 0.2, 0.2], physics=phys))
    # Static table surface proxy (RoboTwin's table top sits at z=0.74).
    objects.append(
        PrimitiveCubeCfg(name="table", size=(0.96, 0.96, 0.74), color=[0.85, 0.85, 0.85], physics=PhysicStateType.GEOM)
    )

    # --observer-cam matches RoboTwin's built-in observer_camera (camera.py) so the
    # RoboVerse render can be composited frame-for-frame with native_render.py output.
    if getattr(args, "observer_cam", False):
        # Matched side-by-side camera: identical pos/look_at/fovy to native_render's
        # roll=0 world-up camera, so the two videos composite frame-for-frame.
        # fovy 55deg on a 512^2 sensor -> focal_length/horizontal_aperture below.
        camera = PinholeCameraCfg(
            name="main_camera", pos=[0.0, 0.6, 1.35], look_at=[0.0, -0.3, 0.78],
            width=512, height=512, focal_length=20.0, horizontal_aperture=20.955, data_types=["rgb"],
        )  # fmt: skip
    else:
        camera = PinholeCameraCfg(
            name="main_camera", pos=[1.3, -0.5, 1.5], look_at=[0.1, -0.2, 0.85],
            width=640, height=480, data_types=["rgb"],
        )  # fmt: skip
    scenario = ScenarioCfg(
        robots=[robot],
        objects=objects,
        cameras=[camera] if args.video else [],
        sim_params=SimParamCfg(dt=0.01),
        decimation=4,
        simulator=args.sim,
        num_envs=1,
        headless=True,
        add_default_ground=True,
    )
    # Ray-traced render to match RoboTwin (_base_task sets the same), a GLOBAL sapien
    # setting -- must run before the renderer/handler is created.
    if getattr(args, "rt", False):
        try:
            import sapien

            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")
        except Exception as e:
            log.warning(f"RT shader unavailable: {e}")

    handler = get_handler(scenario)

    # Articulation objects need a dof_pos in every state set. For a *moving* object
    # (door/lid/drawer) drive its joints from RoboTwin's recorded per-frame qpos
    # (mapped RoboTwin-joint-name -> RoboVerse-joint-order); otherwise hold at zeros.
    rv_to_n = {v: k for k, v in name_map.items()}
    artic_drive = {}  # rv -> (rv_joint_names, cols|None, joint_traj|None)
    for rv in artic_names:
        try:
            rv_jnames = handler.get_joint_names(rv, sort=True)
        except Exception:
            rv_jnames = []
        n = rv_to_n.get(rv)
        if n in moving_artic and object_joint_names.get(n):
            rec = object_joint_names[n]
            cols = [rec.index(j) if j in rec else None for j in rv_jnames]
            artic_drive[rv] = (rv_jnames, cols, moving_artic[n])
        else:
            artic_drive[rv] = (rv_jnames, None, None)

    # Physics mode drives the robot through the canonical get_traj action stream
    # (correctly keyed by robot name), exactly like the joint-parity harness.
    robot_actions = None
    if not kinematic:
        tmp_v2 = "outputs/robotwin_coverage/_mesh_replay_v2.pkl"
        bridge_to_v2(bridge, tmp_v2)
        _, all_actions, _ = get_traj(tmp_v2, robot)
        robot_actions = all_actions[0]

    def _obj_state(t):
        out = {}
        for n in manip:
            tr = object_traj[n]
            p = tr[min(t, len(tr) - 1)]
            rv = name_map[n]
            st = {"pos": [float(x) for x in p[:3]], "rot": [float(x) for x in p[3:7]]}
            if rv in artic_drive:
                jn, cols, jtraj = artic_drive[rv]
                if cols is not None:
                    q = jtraj[min(t, len(jtraj) - 1)]
                    st["dof_pos"] = {j: (float(q[c]) if c is not None else 0.0) for j, c in zip(jn, cols)}
                else:
                    st["dof_pos"] = {j: 0.0 for j in jn}
            out[rv] = st
        # table proxy at RoboTwin's fixed surface
        out["table"] = {"pos": [0.0, 0.0, 0.37], "rot": [1.0, 0.0, 0.0, 0.0]}
        return out

    # Initial state: robot home (achieved frame 0) + objects at frame 0.
    init = {
        "robots": {ROBOT_NAME: {"pos": ROBOT_POS, "rot": ROBOT_ROT, "dof_pos": vector_to_dof(real[0], ls, rs)}},
        "objects": _obj_state(0),
    }
    handler.set_states([init])

    obs_saver = None
    if args.video:
        os.makedirs("outputs/robotwin_coverage", exist_ok=True)
        vp = f"outputs/robotwin_coverage/mesh_replay_{task}_{args.mode}.mp4"
        obs_saver = ObsSaver(video_path=vp)
        obs_saver.add(handler.get_states(mode="tensor"))

    rv_obj_traj = {n: [] for n in manip}
    n_frames = len(vectors)
    for t in range(1, n_frames):
        if kinematic:
            # Teleport robot (achieved qpos) + objects (recorded poses): faithful playback.
            st = {
                "robots": {ROBOT_NAME: {"pos": ROBOT_POS, "rot": ROBOT_ROT, "dof_pos": vector_to_dof(real[t], ls, rs)}},
                "objects": _obj_state(t),
            }
            # Pure playback: teleport to the exact recorded state, then refresh the
            # render WITHOUT a physics step. Running simulate() here makes the PD
            # controller lurch the arm toward stale dof targets every frame -> violent
            # shaking; skipping it entirely leaves the render stale. refresh_render()
            # updates the scene render so the camera sees RoboTwin's exact qpos.
            handler.set_states([st])
            if hasattr(handler, "refresh_render"):
                handler.refresh_render()
        else:
            # Robot driven by command target; objects dynamic (contact only).
            handler.set_dof_targets([robot_actions[t - 1]])
            for _ in range(max(1, args.settle)):
                handler.simulate()
            state = handler.get_states(mode="tensor")
            for n in manip:
                root = state.objects[name_map[n]].root_state[0].detach().cpu().numpy()
                rv_obj_traj[n].append(root[:7])  # [x,y,z, qw,qx,qy,qz]
        if obs_saver is not None:
            obs_saver.add(handler.get_states(mode="tensor"))

    if obs_saver is not None:
        obs_saver.save()
        log.info(f"video -> outputs/robotwin_coverage/mesh_replay_{task}_{args.mode}.mp4")

    result = {"task": task, "mode": args.mode, "frames": n_frames, "objects": {}}
    if not kinematic:
        for n in manip:
            rt = object_traj[n][1:]
            rv = np.asarray(rv_obj_traj[n])
            m = min(len(rt), len(rv))
            if m == 0:
                continue
            pos_err = np.linalg.norm(rv[:m, :3] - rt[:m, :3], axis=1)
            ang_err = np.asarray([_quat_angle(rv[i, 3:7], rt[i, 3:7]) for i in range(m)])
            moved = float(np.linalg.norm(rt[-1, :3] - rt[0, :3]))
            result["objects"][n] = {
                "moved_m": moved,
                "final_pos_err_m": float(pos_err[-1]),
                "max_pos_err_m": float(pos_err.max()),
                "mean_pos_err_m": float(pos_err.mean()),
                "final_ang_err_rad": float(ang_err[-1]),
                "max_ang_err_rad": float(ang_err.max()),
            }
            log.info(
                f"[{task}] {n}: moved {moved:.3f}m | final pos err {pos_err[-1]:.4f}m "
                f"max {pos_err.max():.4f}m | final ang {ang_err[-1]:.3f}rad"
            )
    handler.close()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge", default=os.path.expanduser("~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl"))
    ap.add_argument("--all", action="store_true", help="sweep every *.pkl with object_traj under the bridge dir")
    ap.add_argument("--bridge-dir", default=os.path.expanduser("~/projects/robotwin/data/_rv_bridge"))
    ap.add_argument("--mode", choices=["kinematic", "physics"], default="kinematic")
    ap.add_argument("--sim", default="sapien3")
    ap.add_argument("--settle", type=int, default=8, help="(physics mode) simulate() calls per target")
    ap.add_argument("--video", action="store_true")
    ap.add_argument(
        "--observer-cam", action="store_true", help="render from RoboTwin's observer pose (for side-by-side)"
    )
    ap.add_argument("--rt", action="store_true", help="ray-traced render (matches RoboTwin's RT shader)")
    ap.add_argument("--robotwin-dir", default=os.path.expanduser("~/projects/robotwin"))
    ap.add_argument("--out", default="outputs/robotwin_coverage/object_parity.json")
    args = ap.parse_args(argv)

    if args.all:
        paths = sorted(
            p for p in glob.glob(os.path.join(args.bridge_dir, "*.pkl")) if not os.path.basename(p).startswith("_")
        )
    else:
        paths = [args.bridge]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results = []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            log.warning(f"[skip] missing {p}")
            continue
        with open(p, "rb") as f:
            bridge = pickle.load(f)
        # In --all we only measure parity; skip pickles without an object trajectory.
        if args.all and not bridge.get("object_traj"):
            continue
        try:
            r = _replay_one(bridge, args)
        except Exception as e:
            log.error(f"[{i + 1}/{len(paths)}] {bridge.get('task')} FAILED: {type(e).__name__}: {e}")
            results.append({"task": bridge.get("task"), "mode": args.mode, "error": f"{type(e).__name__}: {e}"})
            continue
        results.append(r)
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "settle": args.settle, "results": results}, f, indent=2)

    if args.mode == "physics":
        # Aggregate: best (smallest final pos err) object per task, count <= thresholds.
        per_task = []
        for r in results:
            errs = [o["final_pos_err_m"] for o in r.get("objects", {}).values()]
            moved = [o["moved_m"] for o in r.get("objects", {}).values()]
            if errs:
                per_task.append((r["task"], max(errs), max(moved) if moved else 0.0))
        le5 = sum(1 for _, e, _ in per_task if e <= 0.05)
        le3 = sum(1 for _, e, _ in per_task if e <= 0.03)
        log.info(f"\n=== OBJECT-POSE PARITY: {len(per_task)} tasks | <=5cm: {le5} | <=3cm: {le3} ===")
        for t, e, mv in sorted(per_task, key=lambda x: x[1]):
            log.info(f"  {e:.4f}m  {t} (object moved {mv:.3f}m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
