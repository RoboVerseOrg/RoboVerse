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
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
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
    manip = [n for n in object_traj if n not in _INFRA and len(object_traj[n])]
    log.info(f"[{task}] {len(vectors)} frames; manipulated objects: {manip}; meshes: {list(object_meshes)}")

    robot = AlohaAgilexCfg()
    if not robot.urdf_path:
        raise FileNotFoundError("ALOHA-AgileX URDF not found; extract RoboTwin embodiments.zip")

    # Build object cfgs: real mesh where we have one, else a small primitive proxy.
    kinematic = args.mode == "kinematic"
    phys = PhysicStateType.XFORM if kinematic else PhysicStateType.GEOM
    objects, name_map = [], {}
    for n in manip:
        rv = _safe(n)
        name_map[n] = rv
        mesh = object_meshes.get(n)
        if mesh:
            mesh_abs = os.path.join(args.robotwin_dir, mesh["visual"])
            urdf = _glb_to_urdf(mesh_abs, mesh.get("scale"), "outputs/robotwin_coverage/_obj_urdf", rv)
            objects.append(RigidObjCfg(name=rv, urdf_path=urdf, physics=phys, fix_base_link=kinematic))
        else:
            objects.append(PrimitiveCubeCfg(name=rv, size=(0.05, 0.05, 0.05), color=[0.8, 0.2, 0.2], physics=phys))
    # Static table surface proxy (RoboTwin's table top sits at z=0.74).
    objects.append(
        PrimitiveCubeCfg(name="table", size=(0.8, 0.8, 0.74), color=[0.7, 0.6, 0.5], physics=PhysicStateType.GEOM)
    )

    camera = PinholeCameraCfg(
        name="main_camera", pos=[1.3, -0.5, 1.5], look_at=[0.1, -0.2, 0.85], width=640, height=480, data_types=["rgb"]
    )
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
    handler = get_handler(scenario)

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
            out[name_map[n]] = {"pos": [float(x) for x in p[:3]], "rot": [float(x) for x in p[3:7]]}
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
            handler.set_states([st])
            handler.simulate()
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
