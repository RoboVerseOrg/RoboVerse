#!/usr/bin/env python3
"""
Replay xhand (or other URDF hand) joint trajectories + HOI4D object
using MetaSim simulator interface (default: PyBullet).

- Use MetaSim to create the simulator (same style as metasim/scripts/replay_demo.py).
- Access the underlying PyBullet client via env.handler.
- Preserve original features: auto-discovery for HOI4D, object pose RDF->FLU,
  optional extrinsics, video saving (headless), GUI camera, etc.
"""

from __future__ import annotations

import argparse
import json
import logging

# os.environ.setdefault("PYBULLET_USE_GUI", "0")
# os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
# os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# os.environ.setdefault("PYBULLET_EGL", "1")
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ====== MetaSim imports (no pybullet import) ======
from loguru import logger as log
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation as R

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import HybridSimEnv
from metasim.utils.setup_util import get_sim_env_class

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


# ---------- Hand traj ----------
def load_npz(npz_path: str | Path):
    data = np.load(str(npz_path), allow_pickle=True)
    qpos = data["qpos"]  # (T, J)
    meta_raw = data["meta"]
    meta = json.loads(meta_raw.decode("utf-8") if isinstance(meta_raw, (bytes, bytearray)) else str(meta_raw))
    fps = int(meta.get("fps", 15))
    width = int(meta.get("width", 640))
    height = int(meta.get("height", 480))
    return qpos, meta, fps, width, height


# ---------- HOI4D object helpers (same as before, small tweaks) ----------
HOI4D_OBJECT_TYPES = {
    "toycar": "rigid",
    "mug": "rigid",
    "laptop": "articulated",
    "storagefurniture": "articulated",
    "bottle": "rigid",
    "safe": "articulated",
    "bowl": "rigid",
    "bucket": "articulated",
    "scissors": "articulated",
    "pliers": "articulated",
    "kettle": "rigid",
    "knife": "rigid",
    "trashcan": "articulated",
    "lamp": "articulated",
    "stapler": "articulated",
    "chair": "rigid",
}

HOI4D_OBJECT_CLASSES = {
    1: "ToyCar",
    2: "Mug",
    3: "Laptop",
    4: "StorageFurniture",
    5: "Bottle",
    6: "Safe",
    7: "Bowl",
    8: "Bucket",
    9: "Scissors",
    11: "Pliers",
    12: "Kettle",
    13: "Knife",
    14: "TrashCan",
    17: "Lamp",
    18: "Stapler",
    20: "Chair",
}
_HOI4D_CLASS_INDEX = [
    "",
    "toycar",
    "mug",
    "laptop",
    "storagefurniture",
    "bottle",
    "safe",
    "bowl",
    "bucket",
    "scissors",
    "",
    "pliers",
    "kettle",
    "knife",
    "trashcan",
    "",
    "",
    "lamp",
    "stapler",
    "",
    "chair",
]


def auto_discover_hoi4d_files(root: Path, capture_name: str):
    rgb = root / "HOI4D_release" / capture_name / "align_rgb" / "image.mp4"
    if not rgb.exists():
        log.warning(f"RGB not found: {rgb}")

    parts = capture_name.split("/")
    if len(parts) != 7:
        raise ValueError(f"capture_name should be 'C01/H01/O02/I01/R01/L01/T01', got: {capture_name}")
    camera_id, human_id, obj_class_id, obj_inst_id, room_id, layout_id, task_id = parts
    class_idx = int(obj_class_id[1:])
    inst_idx = int(obj_inst_id[1:])
    class_name = HOI4D_OBJECT_CLASSES.get(class_idx, None)
    if class_name is None:
        raise RuntimeError(f"Unknown HOI4D class id: {class_idx}")

    obj_file = root / "HOI4D_CAD_Model_for_release" / "rigid" / class_name / f"{inst_idx:03d}.obj"
    if not obj_file.exists():
        class_key = _HOI4D_CLASS_INDEX[class_idx]
        typ = HOI4D_OBJECT_TYPES.get(class_key, "unknown")
        if typ != "rigid":
            raise RuntimeError(f"Class '{class_name}' is {typ} (articulated not supported).")
        raise FileNotFoundError(f"Object .obj not found: {obj_file}")

    objpose_dir = root / "HOI4D_annotations" / capture_name / "objpose"
    if not objpose_dir.exists():
        raise FileNotFoundError(f"objpose dir not found: {objpose_dir}")

    extrinsics_file = root / "HOI4D_annotations" / capture_name / "3Dseg" / "output.log"
    if not extrinsics_file.exists():
        extrinsics_file = None

    return obj_file, objpose_dir, extrinsics_file, rgb


def load_object_pose_from_dir(objpose_dir: Path):
    frames = []
    for pth in sorted(objpose_dir.glob("*.json"), key=lambda p: int(p.stem)):
        with open(pth) as f:
            data = json.load(f)
        if "dataList" in data:
            d = data["dataList"][0]
        else:
            raise NotImplementedError("Multiple objects not supported in this loader.")

        if d.get("label") == "Watercup":
            d["label"] = "Mug"
        if d.get("label") == "bottleddrinks":
            d["label"] = "Bottle"

        rot = d["rotation"]  # euler xyz
        trans = d["center"]
        q_xyzw = R.from_euler("xyz", [rot["x"], rot["y"], rot["z"]]).as_quat()
        t = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float32)
        frames.append(np.concatenate([q_xyzw.astype(np.float32), t], axis=0))  # (7,)
    if not frames:
        raise RuntimeError(f"No frames in {objpose_dir}")
    return np.stack(frames, axis=0)  # (T,7)


def parse_open3d_trajectory_extrinsics(output_log: Path):
    try:
        lines = [ln.strip() for ln in output_log.read_text().splitlines() if ln.strip()]
    except Exception:
        return None

    mats = []
    # Try [[...]] style
    buf = []
    for ln in lines:
        if ln.startswith("["):
            row = ln.replace("[", " ").replace("]", " ").replace(",", " ").split()
            if len(row) >= 4:
                buf.append([float(x) for x in row[:4]])
            if len(buf) == 4:
                mats.append(np.array(buf, dtype=np.float32))
                buf = []
    if mats:
        return np.stack(mats, axis=0)

    # Try numeric blocks (idx idx idx + 4 rows of 4x4)
    mats = []
    i, n = 0, len(lines)
    while i < n:
        parts = lines[i].split()
        if len(parts) == 3 and all(p.replace("-", "").isdigit() for p in parts):
            i += 1
            if i + 3 >= n:
                break
            try:
                M = []
                for _ in range(4):
                    row = [float(x) for x in lines[i].split()]
                    if len(row) < 4:
                        raise ValueError
                    M.append(row[:4])
                    i += 1
                mats.append(np.array(M, dtype=np.float32))
                continue
            except Exception:
                pass
        i += 1

    if mats:
        return np.stack(mats, axis=0)
    return None


def transform_rdf_to_flu(R_c, t_c):
    T = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    return T @ R_c @ T.T, T @ t_c


def apply_extra_correction(R_w, euler_deg: Optional[str] = None):
    if not euler_deg:
        return R_w
    yaw, pitch, roll = [float(x) for x in euler_deg.split(",")]
    R_corr = R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix().astype(np.float32)
    return R_w @ R_corr


# ---------- helpers that DO NOT import pybullet ----------
# def _get_pb_client(handler):
#     """
#     Try to fetch the underlying pybullet client from MetaSim handler.
#     """
#     # Common possibilities across handlers:
#     for attr in ("p", "client", "bullet", "pb", "_client", "_p"):
#         if hasattr(handler, attr):
#             return getattr(handler, attr)
#     raise RuntimeError("Cannot locate PyBullet client from env.handler; "
#                        "please check MetaSim PyBullet handler implementation.")


def _get_pb_client(handler):
    """
    Return the pybullet *module* so we can call loadURDF/getCameraImage/etc.
    Do NOT return handler.client (that's just an int connection id).
    """
    # try:
    #     import pybullet as p
    #     return p
    # except Exception:
    #     pass
    # fallback: try to find a module-like attribute but avoid 'client'
    for attr in ("p", "pb", "bullet", "_p"):
        if hasattr(handler, attr):
            cand = getattr(handler, attr)
            # must look like the pybullet module
            if hasattr(cand, "loadURDF") and hasattr(cand, "getCameraImage"):
                return cand
    raise RuntimeError("Cannot locate a pybullet API; expected module-like object with loadURDF().")


def _get_active_joint_indices(pb, body_id):
    n = pb.getNumJoints(body_id)
    idx, names = [], []
    for j in range(n):
        info = pb.getJointInfo(body_id, j)
        joint_type = info[2]
        name = info[1].decode("utf-8")
        if joint_type != pb.JOINT_FIXED:
            idx.append(j)
            names.append(name)
    return idx, names


# ---------- main ----------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    ap.add_argument("--urdf", type=str, required=True, help="Path to xhand_right*.urdf")
    ap.add_argument("--gui", action="store_true", help="Use GUI (default: headless)")
    ap.add_argument("--save-mp4", type=str, default=None, help="Path to save mp4 when headless")
    ap.add_argument("--fps", type=int, default=None, help="Override playback FPS")
    ap.add_argument("--scale", type=float, default=1.0, help="URDF globalScaling")
    ap.add_argument("--ee-cam", action="store_true", help="Attach camera to hand base link for debugging")
    ap.add_argument("--include-dummy", action="store_true", help="Drive dummy_* base joints if present")
    ap.add_argument("--print-mapping", action="store_true", help="Print name mapping and limits")
    ap.add_argument("--offset-first", action="store_true", help="Subtract first frame qpos")
    ap.add_argument("--clamp", action="store_true", help="Clamp to URDF limits")
    ap.add_argument("--flip", type=str, default="", help="Comma-separated joint names to flip sign")

    # HOI4D options
    ap.add_argument("--hoi4d-root", type=str, default=None, help="miniHOI4D root for auto discovery")
    ap.add_argument("--capture", type=str, default=None, help="Capture like C01/H01/O02/I01/R01/L01/T01")
    ap.add_argument("--obj-mesh", type=str, default=None, help="Manual: path to object .obj")
    ap.add_argument("--objpose-dir", type=str, default=None, help="Manual: path to objpose/*.json dir")
    ap.add_argument("--extrinsics-file", type=str, default=None, help="Manual: 3Dseg/output.log path")
    ap.add_argument("--use_extrinsics", action="store_true", help="Use camera->world extrinsics to place object")
    ap.add_argument("--obj-scale", type=float, default=1.0, help="Scale for object mesh")
    ap.add_argument(
        "--obj-corr-euler", type=str, default=None, help='Extra orientation correction "yaw,pitch,roll" (deg)'
    )

    # MetaSim options (default choose pybullet)
    ap.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"],
    )
    ap.add_argument(
        "--renderer",
        type=str,
        default=None,
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Headless rendering if supported; for pybullet we still use TinyRenderer in offscreen.",
    )
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    # ---- Load hand traj ----
    qpos, meta, file_fps, width, height = load_npz(args.npz)
    fps = args.fps or file_fps
    dt = 1.0 / fps

    # ---- Discover or manual object files ----
    if args.obj_mesh and args.objpose_dir:
        obj_mesh = Path(args.obj_mesh).expanduser().absolute()
        objpose_dir = Path(args.objpose_dir).expanduser().absolute()
        extrinsics_file = Path(args.extrinsics_file).expanduser().absolute() if args.extrinsics_file else None
        if not obj_mesh.exists():
            raise FileNotFoundError(obj_mesh)
        if not objpose_dir.exists():
            raise FileNotFoundError(objpose_dir)
    elif args.hoi4d_root and args.capture:
        root = Path(args.hoi4d_root).expanduser().absolute()
        obj_mesh, objpose_dir, extrinsics_file_auto, rgb = auto_discover_hoi4d_files(root, args.capture)
        extrinsics_file = extrinsics_file_auto
        if (width == 0 or height == 0) and (rgb is not None and rgb.exists()):
            try:
                cap = cv2.VideoCapture(str(rgb))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
            except Exception:
                pass
    else:
        raise ValueError(
            "Please select: Automatic detection (--hoi4d-root + --capture) or Manual specification (--obj-mesh + --objpose-dir)."
        )

    # ---- Load object pose ----
    obj_pose_cam = load_object_pose_from_dir(objpose_dir)  # (T,7) [xyzw, t]
    T_obj = obj_pose_cam.shape[0]

    # ---- Optionally parse extrinsics ----
    extrinsics_seq = None
    if args.use_extrinsics and extrinsics_file is not None and Path(extrinsics_file).exists():
        extrinsics_seq = parse_open3d_trajectory_extrinsics(Path(extrinsics_file))
        if extrinsics_seq is not None and len(extrinsics_seq) != T_obj:
            L = min(len(extrinsics_seq), T_obj)
            extrinsics_seq = extrinsics_seq[:L]
            obj_pose_cam = obj_pose_cam[:L]
            T_obj = L
        if extrinsics_seq is None:
            log.warning(f"Failed to parse extrinsics from {extrinsics_file}, fallback to RDF->FLU only.")

    # =========================================================
    # ============ Launch simulator via MetaSim ===============
    # =========================================================
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(
        task=None,  # no built-in task
        robots=[],  # we load URDF hand ourselves through pybullet client
        scene=None,
        cameras=[camera],
        random=RandomizationCfg(),
        render=RenderCfg(),
        sim=args.sim,
        renderer=args.renderer,
        num_envs=1,
        try_add_table=False,
        object_states=False,
        split="all",
        headless=args.headless or (not args.gui),
    )

    if scenario.renderer is None:
        log.info(f"Using simulator: {scenario.sim}")
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
    else:
        log.info(f"Using simulator: {scenario.sim}, renderer: {scenario.renderer}")
        env_class_render = get_sim_env_class(SimType(scenario.renderer))
        env_render = env_class_render(scenario)
        env_class_physics = get_sim_env_class(SimType(scenario.sim))
        env_physics = env_class_physics(scenario)
        env = HybridSimEnv(env_physics, env_render)

    handler = env.handler
    pb = _get_pb_client(handler)  # <-- pybullet client from handler

    # Configure physics
    try:
        pb.setGravity(0, 0, 0)
    except Exception:
        pass
    try:
        pb.setTimeStep(dt)
    except Exception:
        pass

    # ---- Load hand URDF (absolute path recommended) ----
    urdf_path = str(Path(args.urdf).expanduser().absolute())
    hand_id = pb.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,
        flags=getattr(pb, "URDF_USE_SELF_COLLISION", 0),
        globalScaling=args.scale,
    )

    # ---- Active joints & limits ----
    active_idx, urdf_joint_names = _get_active_joint_indices(pb, hand_id)
    print(f"[Info] URDF movable joints ({len(active_idx)}): {urdf_joint_names}")

    limits = {}
    for j, name in zip(active_idx, urdf_joint_names):
        info = pb.getJointInfo(hand_id, j)
        lo, hi = float(info[8]), float(info[9])
        limits[name] = (lo, hi)

    # ---- Name mapping ----
    meta_names_full = meta.get("joint_names", None) or meta.get("robot_joint_names", None)
    col_idx_for_q = None
    driven_names = None
    if meta_names_full is not None:
        meta_names_full = [str(x) for x in meta_names_full]
        meta_names_filtered = (
            meta_names_full[:] if args.include_dummy else [n for n in meta_names_full if not n.startswith("dummy_")]
        )
        name2idx = {jn: j for j, jn in zip(active_idx, urdf_joint_names)}
        name2col = {n: i for i, n in enumerate(meta_names_full)}
        mapped_idx, col_idx_for_q, missing = [], [], []
        for jn in meta_names_filtered:
            if jn in name2idx and jn in name2col:
                mapped_idx.append(name2idx[jn])
                col_idx_for_q.append(name2col[jn])
            else:
                missing.append(jn)
        if mapped_idx:
            active_idx = mapped_idx
            driven_names = [meta_names_full[c] for c in col_idx_for_q]
            print(f"[Info] remapped by names. Driving {len(active_idx)} joints.")
            if missing:
                print(f"[Warn] joints in npz not found in URDF: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        else:
            print("[Warn] name map failed; fallback to index based.")
            col_idx_for_q = None
    else:
        print("[Warn] no joint_names in npz; use index-based min alignment.")

    if args.print_mapping and col_idx_for_q is not None:
        print("---- Mapping (npz -> URDF) ----")
        q0 = qpos[0]
        for npz_col, j in zip(col_idx_for_q, active_idx):
            npz_name = driven_names[col_idx_for_q.index(npz_col)] if driven_names else "(unknown)"
            lo, hi = limits.get(npz_name, (-np.inf, np.inf))
            print(f"{npz_name:30s} -> idx {j:2d} | q0={q0[npz_col]: .4f} | lim=({lo: .3f},{hi: .3f})")
        print("--------------------------------")

    # ---- Create object (visual + collision) ----
    obj_visual = pb.createVisualShape(
        shapeType=pb.GEOM_MESH,
        fileName=str(obj_mesh),
        meshScale=[args.obj_scale] * 3,
        rgbaColor=[0.9, 0.9, 0.9, 1.0],
    )
    obj_collision = pb.createCollisionShape(
        shapeType=pb.GEOM_MESH,
        fileName=str(obj_mesh),
        meshScale=[args.obj_scale] * 3,
    )
    obj_id = pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=obj_collision,
        baseVisualShapeIndex=obj_visual,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )

    # ---- Video writer ----
    writer = None
    if (not args.gui) and args.save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_mp4, fourcc, fps, (max(1, width), max(1, height)))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {args.save_mp4}")
    elif (not args.gui) and (not args.save_mp4):
        log.warning("Headless mode but --save-mp4 not set; nothing will be saved.")

    # ---- GUI camera (pybullet debug view) ----
    if args.gui:
        try:
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=[0.3, 0, 0.1]
            )
        except Exception:
            pass

    # ---- Hand playback helpers ----
    flip_set = {s.strip() for s in args.flip.split(",") if s.strip()} if args.flip else set()

    q0_offset = None
    if args.offset_first:
        if col_idx_for_q is not None:
            q0_offset = qpos[0, col_idx_for_q].astype(float)
        else:
            q0_offset = qpos[0, : min(qpos.shape[1], len(active_idx))].astype(float)

    # ---- Main loop ----
    T = qpos.shape[0]
    t0 = time.time()
    for i in range(min(T, T_obj)):
        # ------ Hand joints ------
        if col_idx_for_q is not None:
            q = qpos[i, col_idx_for_q].astype(float)
        else:
            dim = min(qpos.shape[1], len(active_idx))
            q = qpos[i, :dim].astype(float)

        if q0_offset is not None and len(q0_offset) == len(q):
            q = q - q0_offset
        if flip_set and driven_names is not None and len(driven_names) == len(q):
            for k, name in enumerate(driven_names):
                if name in flip_set:
                    q[k] = -q[k]
        if args.clamp:
            names_for_clamp = driven_names if driven_names is not None else urdf_joint_names[: len(q)]
            for k, name in enumerate(names_for_clamp):
                lo, hi = limits.get(name, (-1e10, 1e10))
                if lo <= hi:
                    if q[k] < lo:
                        q[k] = lo
                    if q[k] > hi:
                        q[k] = hi
        for k, j in enumerate(active_idx[: len(q)]):
            pb.resetJointState(hand_id, j, float(q[k]))

        # ------ Object pose ------
        qxyzw = obj_pose_cam[i, :4]
        t_cam = obj_pose_cam[i, 4:]
        R_cam = R.from_quat(qxyzw).as_matrix().astype(np.float32)  # camera (RDF)
        R_w, t_w = transform_rdf_to_flu(R_cam, t_cam)  # -> world (FLU / Z-up)
        R_w = apply_extra_correction(R_w, args.obj_corr_euler)

        if args.use_extrinsics and extrinsics_seq is not None:
            E = extrinsics_seq[i]  # (4,4) camera->world (Open3D)
            M_cam = np.eye(4, dtype=np.float32)
            M_cam[:3, :3] = R_cam
            M_cam[:3, 3] = t_cam
            M_world = E @ M_cam
            R_w2, t_w2 = transform_rdf_to_flu(M_world[:3, :3], M_world[:3, 3])
            R_w, t_w = R_w2, t_w2

        q_w_xyzw = R.from_matrix(R_w).as_quat().astype(np.float32)  # xyzw
        q_w_wxyz = np.array([q_w_xyzw[3], q_w_xyzw[0], q_w_xyzw[1], q_w_xyzw[2]], dtype=np.float32)
        pb.resetBasePositionAndOrientation(obj_id, t_w.tolist(), q_w_wxyz.tolist())

        # ------ Step simulation ------
        try:
            pb.stepSimulation()
        except Exception:
            pass

        # ------ Render & save (offscreen) ------
        if writer is not None:
            if args.ee_cam:
                base_pos, _ = pb.getBasePositionAndOrientation(hand_id)
                view = pb.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=base_pos, distance=0.6, yaw=180, pitch=-30, roll=0, upAxisIndex=2
                )
            else:
                view = pb.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0.3, 0, 0.1], distance=1.0, yaw=180, pitch=-30, roll=0, upAxisIndex=2
                )
            aspect = float(width) / float(height) if height > 0 else 1.0
            proj = pb.computeProjectionMatrixFOV(fov=60, aspect=aspect, nearVal=0.01, farVal=5.0)
            _w, _h, rgb, _depth, _seg = pb.getCameraImage(
                max(1, width),
                max(1, height),
                view,
                proj,
                renderer=getattr(pb, "ER_BULLET_HARDWARE_OPENGL", pb.ER_TINY_RENDERER),
            )
            rgb = np.asarray(rgb)[..., :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if args.gui:
            # simple real-time pacing
            time.sleep(max(0.0, t0 + (i + 1) * dt - time.time()))

        if (i + 1) % 30 == 0:
            print(f"frame {i + 1}/{min(T, T_obj)}")

    if writer is not None:
        writer.release()
        print(f"[Saved] {args.save_mp4}")

    # Clean up env
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
