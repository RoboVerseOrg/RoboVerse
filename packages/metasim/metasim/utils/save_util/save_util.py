"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import json
import os
import pickle as pkl

import imageio as iio
import numpy as np
import torch
from loguru import logger as log

from metasim.types import DictEnvState
from metasim.utils.io_util import write_16bit_depth_video
from metasim.utils.kinematics import get_ee_state_from_list
from metasim.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat, quat_apply


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def _camera_metadata(camera_state: dict) -> dict[str, list]:
    """The ``cam_*`` metadata of one frame from a nested camera dict (``state_tensor_to_nested``).

    ``cam_pos`` is the camera position, ``cam_intr`` the 3x3 intrinsics, ``cam_extr`` the 4x4 world-to-camera
    transform in the OpenCV convention (x right, y down, z forward), the matrix ``camera_util.get_cam_params``
    builds and ``obs_utils.get_pcd_from_rgbd`` consumes, and ``cam_look_at`` a point on the optical axis one
    metre ahead of the camera. Only ``cam_extr`` carries the roll of a mounted camera: rebuilding the extrinsic
    from ``cam_pos`` / ``cam_look_at`` with ``get_cam_params`` assumes world +Z is up. Each key is written from the field it needs (``cam_pos`` from ``pos``, ``cam_intr`` from
    ``intrinsics``, the other two from ``pos`` and ``quat_world``) and is an empty list when the backend leaves
    that field None, as every entry was before the pose travelled with the state.
    """
    pos = camera_state.get("pos")
    quat = camera_state.get("quat_world")
    intrinsics = camera_state.get("intrinsics")
    out = {
        "cam_pos": [] if pos is None else torch.as_tensor(pos).tolist(),
        "cam_intr": [] if intrinsics is None else torch.as_tensor(intrinsics).tolist(),
        "cam_look_at": [],
        "cam_extr": [],
    }
    if pos is not None and quat is not None:
        quat = torch.as_tensor(quat, dtype=torch.float32).unsqueeze(0)
        pos = torch.as_tensor(pos, dtype=torch.float32)
        forward = quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]]))[0]  # +X is forward in the world convention
        c2w_cv = matrix_from_quat(convert_camera_frame_orientation_convention(quat, origin="world", target="ros"))[0]
        w2c = torch.eye(4)
        w2c[:3, :3] = c2w_cv.T
        w2c[:3, 3] = -c2w_cv.T @ pos
        out["cam_extr"] = w2c.tolist()
        out["cam_look_at"] = (pos + forward).tolist()
    return out


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

        for key, value in _camera_metadata(camera_state).items():
            metadata[key].append(value)

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
    ee_states = get_ee_state_from_list(demo, robot_config, tensorize=True)  # (T, 7): pos, rpy, gripper
    if len(demo) and ee_states.shape[0] == 0:
        log.warning(
            f"{save_dir}: 'ee_state' is empty; this recording carries no body states (the backend reports none), "
            "so no end-effector pose can be derived. Converters that need it will refuse this demo."
        )
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
