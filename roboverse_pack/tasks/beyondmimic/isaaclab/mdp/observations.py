# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: import paths rewired to RoboVerse; env type hints retargeted to TrackingRLEnv; docstrings and explanatory comments added.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.isaaclab.envs.tracking_rl_env import TrackingRLEnv
    from roboverse_pack.tasks.beyondmimic.isaaclab.mdp.commands import MotionCommand


def robot_anchor_ori_w(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Robot anchor orientation in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Robot anchor linear velocity in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Robot anchor angular velocity in world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


# NOTE observation callback results will be concatenated into a single tensor
def robot_body_pos_b(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Body positions relative to (robot) anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )  # [n_envs, n_bodies, 3] positions of each body relative to the anchor frame

    return pos_b.view(env.num_envs, -1)  # [n_envs, n_bodies * 3]


def robot_body_ori_b(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Body orientations relative to anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Target anchor position relative to anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: TrackingRLEnv, command_name: str) -> torch.Tensor:
    """Target anchor orientation relative to anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,  # [n_envs, 3]
        command.robot_anchor_quat_w,  # [n_envs, 4]
        command.anchor_pos_w,  # [n_envs, 3]
        command.anchor_quat_w,  # [n_envs, 4]
    )  # [n_envs, 4] quaternion representing the relative rotation between the two frames
    mat = matrix_from_quat(ori)  # [n_envs, 3, 3] convert to rotation matrix
    return mat[..., :2].reshape(
        mat.shape[0], -1
    )  # [n_envs, 6] extract the first two rows because the third row can be derived from orthogonality
