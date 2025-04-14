"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import json
import os
import pickle as pkl

import imageio as iio
import numpy as np
import torch

from metasim.types import EnvState
from metasim.utils.io_util import write_16bit_depth_video


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def is_v2_demo(demo: list[EnvState]) -> bool:
    """Check if the demo is in the v2 format. Demo should be v3 state format."""
    return "robots" in demo[0]


def save_demo_v2(save_dir: str, demo: list[EnvState]):
    """Save a demo to a directory.

    Args:
        save_dir: The directory to save the demo.
        demo: The demo to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get the main robot name (assuming first robot in first state)
    robot_name = next(iter(demo[0]["robots"].keys()))
    # Get the main camera name (assuming first camera in first state)
    camera_name = next(iter(demo[0]["cameras"].keys()))

    # Convert and prepare data for saving
    rgbs = []
    depths = []
    jsondata = {
        "depth_min": [],
        "depth_max": [],
        "cam_pos": [],
        "cam_look_at": [],
        "cam_intr": [],
        "cam_extr": [],
        "joint_qpos_target": [],
        "joint_qpos": [],
        "robot_ee_state": [],
        "robot_ee_state_target": [],
        "robot_root_state": [],
        "robot_body_state": [],
    }

    # Process each timestep
    for i, state in enumerate(demo):
        # Extract robot state
        robot_state = state["robots"][robot_name]
        camera_state = state["cameras"][camera_name]

        # Extract vision data
        if "rgb" in camera_state:
            rgb = camera_state["rgb"].cpu().numpy()
            rgbs.append(rgb)

        if "depth" in camera_state:
            depth = camera_state["depth"].cpu().numpy()
            depths.append(_normalize_depth(depth))
            jsondata["depth_min"].append(depth.min().item())
            jsondata["depth_max"].append(depth.max().item())

        # Extract camera data
        jsondata["cam_pos"].append(camera_state["cam_pos"].tolist() if "cam_pos" in camera_state else [])
        jsondata["cam_look_at"].append(camera_state["cam_look_at"].tolist() if "cam_look_at" in camera_state else [])
        jsondata["cam_intr"].append(camera_state["cam_intr"].tolist() if "cam_intr" in camera_state else [])
        jsondata["cam_extr"].append(camera_state["cam_extr"].tolist() if "cam_extr" in camera_state else [])

        # Extract robot data
        jsondata["joint_qpos"].append([robot_state["dof_pos"][k] for k in sorted(robot_state["dof_pos"].keys())])

        # For targets, handle them in the same way as the original function
        if i < len(demo) - 1:
            next_robot_state = demo[i + 1]["robots"][robot_name]
            target_dof_pos = [
                next_robot_state["dof_pos_target"][k] for k in sorted(next_robot_state["dof_pos_target"].keys())
            ]
        else:
            # For the last timestep, use the same target as the current state
            target_dof_pos = [robot_state["dof_pos_target"][k] for k in sorted(robot_state["dof_pos_target"].keys())]

        jsondata["joint_qpos_target"].append(target_dof_pos)

        # Extract EE state (position, rotation, last joint)
        # Assuming ee_state is available in the state
        ee_pos = robot_state["pos"]
        ee_rot = robot_state["rot"]
        last_joint_pos = robot_state["dof_pos"][sorted(robot_state["dof_pos"].keys())[-1]]
        robot_ee_state = torch.cat([ee_pos, ee_rot, torch.tensor([last_joint_pos], device=ee_pos.device)]).tolist()
        jsondata["robot_ee_state"].append(robot_ee_state)

        # Set robot_ee_state_target
        if i < len(demo) - 1:
            next_robot_state = demo[i + 1]["robots"][robot_name]
            next_ee_pos = next_robot_state["pos"]
            next_ee_rot = next_robot_state["rot"]
            next_last_joint_pos = next_robot_state["dof_pos"][sorted(next_robot_state["dof_pos"].keys())[-1]]
            next_robot_ee_state = torch.cat([next_ee_pos, next_ee_rot, torch.tensor([next_last_joint_pos])]).tolist()
            jsondata["robot_ee_state_target"].append(next_robot_ee_state)
        else:
            jsondata["robot_ee_state_target"].append(robot_ee_state)

        # Extract root and body state
        jsondata["robot_root_state"].append(
            torch.cat([robot_state["pos"], robot_state["rot"], robot_state["vel"], robot_state["ang_vel"]]).tolist()
        )

    # Save video files
    if rgbs:
        iio.mimsave(os.path.join(save_dir, "rgb.mp4"), rgbs, fps=30, quality=10)

    if depths:
        write_16bit_depth_video(os.path.join(save_dir, "depth_uint16.mkv"), depths, fps=30)
        iio.mimsave(
            os.path.join(save_dir, "depth_uint8.mp4"),
            [(depth * 255).astype(np.uint8) for depth in depths],
            fps=30,
            quality=10,
        )

    # Save metadata
    json.dump(jsondata, open(os.path.join(save_dir, "metadata.json"), "w"))
    pkl.dump(jsondata, open(os.path.join(save_dir, "metadata.pkl"), "wb"))

    # Mark as finished
    with open(os.path.join(save_dir, "status.txt"), "w+") as f:
        f.write("success")
