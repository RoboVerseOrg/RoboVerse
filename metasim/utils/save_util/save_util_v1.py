"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import json
import os
import pickle as pkl

import imageio as iio
import numpy as np
import torch

from metasim.utils.io_util import write_16bit_depth_video


def is_v1_demo(demo: list[dict[str, torch.Tensor]]) -> bool:
    """Check if the demo is in the v1 (deprecated) format."""
    return "robot_ee_state" in demo[0]


def _shift_actions_to_previous_timestep(demo: list[dict[str, torch.Tensor]]):
    for i in range(len(demo) - 1):
        demo[i]["joint_qpos_target"] = demo[i + 1]["joint_qpos_target"]
        demo[i]["robot_ee_state_target"] = demo[i + 1]["robot_ee_state"]
    demo[-1]["robot_ee_state_target"] = demo[-1]["robot_ee_state"]
    return demo


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def save_demo_v1(save_dir: str, demo: list[dict[str, torch.Tensor]]):
    """Save a demo to a directory.

    Args:
        save_dir: The directory to save the demo.
        demo: The demo to save.
    """
    ## TODO: This function needs to be updated to support rlds format
    demo = _shift_actions_to_previous_timestep(demo)
    os.makedirs(save_dir, exist_ok=True)
    rgbs = [data_dict["rgb"].numpy() for data_dict in demo]
    iio.mimsave(os.path.join(save_dir, "rgb.mp4"), rgbs, fps=30, quality=10)
    depths = [_normalize_depth(data_dict["depth"].numpy()) for data_dict in demo]
    write_16bit_depth_video(os.path.join(save_dir, "depth_uint16.mkv"), depths, fps=30)
    iio.mimsave(
        os.path.join(save_dir, "depth_uint8.mp4"),
        [(depth * 255).astype(np.uint8) for depth in depths],
        fps=30,
        quality=10,
    )

    jsondata = {
        ## Vision
        # TODO: support multiple cameras
        "depth_min": [d["depth"].min().item() for d in demo],
        "depth_max": [d["depth"].max().item() for d in demo],
        ## Camera
        "cam_pos": [d["cam_pos"].tolist() for d in demo],
        "cam_look_at": [d["cam_look_at"].tolist() for d in demo],
        "cam_intr": [d["cam_intr"].tolist() for d in demo],  # align with old version
        "cam_extr": [d["cam_extr"].tolist() for d in demo],  # align with old version
        ## Action
        # TODO: missing `ee_state` (`surr_ee_state`) in old version
        "joint_qpos_target": [d["joint_qpos_target"].tolist() for d in demo],
        "joint_qpos": [d["joint_qpos"].tolist() for d in demo],  # align with old version
        "robot_ee_state": [d["robot_ee_state"].tolist() for d in demo],  # align with old version
        "robot_ee_state_target": [d["robot_ee_state_target"].tolist() for d in demo],
        "robot_root_state": [d["robot_root_state"].tolist() for d in demo],
        "robot_body_state": [d["robot_body_state"].tolist() for d in demo],
    }

    json.dump(jsondata, open(os.path.join(save_dir, "metadata.json"), "w"))
    pkl.dump(jsondata, open(os.path.join(save_dir, "metadata.pkl"), "wb"))

    ## mark a finished file
    with open(os.path.join(save_dir, "status.txt"), "w+") as f:
        f.write("success")
