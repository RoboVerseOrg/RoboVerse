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
from metasim.utils.kinematics_utils import get_ee_state_from_list


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def save_demo(save_dir: str, demo: list[DictEnvState], robot_config, task_desc=""):
    """Save a list-state demo sequence and metadata (incl. full EE states)."""
    os.makedirs(save_dir, exist_ok=True)

    robot_name = next(iter(demo[0]["robots"].keys()))
    camera_name = next(iter(demo[0]["cameras"].keys()))

    rgb_frames = []
    depth_frames = []
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

    # Full EE state only (no separate pos/quat/gripper fields)
    ee_states = get_ee_state_from_list(demo, robot_config, tensorize=True)  # (T, 8)
    metadata["ee_state"] = ee_states.detach().cpu().tolist()
    metadata["task_desc"] = task_desc

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
