"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import json
import os
import pickle as pkl

import imageio as iio
import numpy as np
import torch

from metasim.types import DictEnvState
from metasim.utils.io_util import write_16bit_depth_video
from metasim.utils.kinematics import get_ee_state_from_list


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def save_demo(save_dir: str, demo: list[DictEnvState], robot_config, task_desc=""):
    """Save a list-state demo sequence and metadata (incl. full EE states)."""
    os.makedirs(save_dir, exist_ok=True)

    robot_name = next(iter(demo[0]["robots"].keys()))
    camera_name = next(iter(demo[0]["cameras"].keys()))

    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    rgb_frames = []
    depth_frames = []
    # Per-frame heights (stored separately as text, one JSON per frame)
    robot_root_z: list[float] = []
    objects_z: list[dict[str, float]] = []
    metadata = {
        "depth_min": [],
        "depth_max": [],
        "cam_pos": [],
        "cam_look_at": [],
        "cam_intr": [],
        "cam_extr": [],
        "joint_qpos_target": [],
        "joint_qpos": [],
        "robot_root_state": [],
        "ee_state": [],
        "task_desc": [],
    }

    # Save per-frame object states separately as a txt file (one JSON per line).
    # This avoids bloating metadata.{json,pkl} while keeping debugging easy.
    objects_state_txt_path = os.path.join(save_dir, "objects_state.txt")

    with open(objects_state_txt_path, "w") as obj_f:
        obj_f.write('# one JSON per line: {"t": int, "objects": {obj_name: {pos,rot,vel,ang_vel,dof_pos?}}}\n')

        for t, env_state in enumerate(demo):
            robot_state = env_state["robots"][robot_name]
            camera_state = env_state["cameras"][camera_name]

            if "rgb" in camera_state:
                rgb_frames.append(camera_state["rgb"].cpu().numpy())
            if "depth" in camera_state:
                depth_np = camera_state["depth"].cpu().numpy()
                depth_frames.append(_normalize_depth(depth_np))
                metadata["depth_min"].append(float(depth_np.min()))
                metadata["depth_max"].append(float(depth_np.max()))

            metadata["cam_pos"].append(camera_state.get("cam_pos", []).tolist() if "cam_pos" in camera_state else [])
            metadata["cam_look_at"].append(
                camera_state.get("cam_look_at", []).tolist() if "cam_look_at" in camera_state else []
            )
            metadata["cam_intr"].append(camera_state.get("cam_intr", []).tolist() if "cam_intr" in camera_state else [])
            metadata["cam_extr"].append(camera_state.get("cam_extr", []).tolist() if "cam_extr" in camera_state else [])

            metadata["joint_qpos"].append([robot_state["dof_pos"][k] for k in sorted(robot_state["dof_pos"].keys())])

            if next(iter(demo[0]["robots"].values())).get("dof_pos_target", None) is not None:
                if t < len(demo) - 1:
                    next_robot_state = demo[t + 1]["robots"][robot_name]
                    target_dof_pos = [
                        next_robot_state["dof_pos_target"][k] for k in sorted(next_robot_state["dof_pos_target"].keys())
                    ]
                else:
                    target_dof_pos = [
                        robot_state["dof_pos_target"][k] for k in sorted(robot_state["dof_pos_target"].keys())
                    ]
            else:
                target_dof_pos = None
            metadata["joint_qpos_target"].append(target_dof_pos)

            root_state_flat = torch.cat([
                robot_state["pos"],
                robot_state["rot"],
                robot_state["vel"],
                robot_state["ang_vel"],
            ])
            metadata["robot_root_state"].append(root_state_flat.tolist())
            # Record robot base height (z)
            robot_root_z.append(float(robot_state["pos"][2].detach().cpu().item()))

            # Save per-frame object states to txt
            frame_obj_state = {}
            frame_obj_z: dict[str, float] = {}
            for obj_name, obj_state in (env_state.get("objects", {}) or {}).items():
                entry = {
                    "pos": _to_list(obj_state.get("pos", [])),
                    "rot": _to_list(obj_state.get("rot", [])),
                    "vel": _to_list(obj_state.get("vel", [])),
                    "ang_vel": _to_list(obj_state.get("ang_vel", [])),
                }
                # record object height (z) if available
                pos = obj_state.get("pos", None)
                if isinstance(pos, torch.Tensor) and pos.numel() >= 3:
                    frame_obj_z[obj_name] = float(pos[2].detach().cpu().item())
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    frame_obj_z[obj_name] = float(pos[2])
                if "dof_pos" in obj_state and obj_state["dof_pos"] is not None:
                    entry["dof_pos"] = obj_state["dof_pos"]
                frame_obj_state[obj_name] = entry
            obj_f.write(json.dumps({"t": t, "objects": frame_obj_state}, ensure_ascii=False) + "\n")
            objects_z.append(frame_obj_z)

    # Full EE state only (no separate pos/quat/gripper fields)
    ee_states = get_ee_state_from_list(demo, robot_config, tensorize=True)  # (T, 8)
    metadata["ee_state"] = ee_states.detach().cpu().tolist()
    metadata["task_desc"] = task_desc

    # Save per-frame heights (franka + objects) separately as text.
    # ee_states convention here is (x,y,z,...) so z is index 2.
    heights_txt_path = os.path.join(save_dir, "heights.txt")
    try:
        ee_z = ee_states[:, 2].detach().cpu().tolist() if isinstance(ee_states, torch.Tensor) else [None] * len(demo)
    except Exception:
        ee_z = [None] * len(demo)

    with open(heights_txt_path, "w") as f:
        f.write('# one JSON per line: {"t": int, "franka_root_z": float, "ee_z": float|null, "objects_z": {name: z}}\n')
        T = len(demo)
        for t in range(T):
            f.write(
                json.dumps(
                    {
                        "t": t,
                        "franka_root_z": robot_root_z[t] if t < len(robot_root_z) else None,
                        "ee_z": ee_z[t] if t < len(ee_z) else None,
                        "objects_z": objects_z[t] if t < len(objects_z) else {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if rgb_frames:
        iio.mimsave(os.path.join(save_dir, "rgb.mp4"), rgb_frames, fps=30, quality=10)
    if depth_frames:
        write_16bit_depth_video(os.path.join(save_dir, "depth_uint16.mkv"), depth_frames, fps=30)
        iio.mimsave(
            os.path.join(save_dir, "depth_uint8.mp4"),
            [(d * 255).astype(np.uint8) for d in depth_frames],
            fps=30,
            quality=10,
        )

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pkl.dump(metadata, f)

    with open(os.path.join(save_dir, "status.txt"), "w") as f:
        f.write("success")
