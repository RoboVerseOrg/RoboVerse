# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: upstream's observation terms ported from Isaac Lab manager terms to MetaSim's TensorState-based callbacks; body ordering kept as upstream for checkpoint compatibility.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from metasim.types import TensorState
from metasim.utils.math import matrix_from_quat, quat_rotate_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.metasim.envs.base_legged_robot import LeggedRobotTask


# NOTE all return values are in the original order to align with checkpoints trained by original BeyondMimic repo


def robot_body_pos_b(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Body positions relative to (robot) anchor frame."""
    robot_state = env_states.robots[env.name]
    body_indexes = env.commands.body_indexes
    anchor_index = env.commands.robot_anchor_body_index

    num_bodies = len(env.commands.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        robot_state.body_state[:, anchor_index : anchor_index + 1, :3].repeat(1, num_bodies, 1),
        robot_state.body_state[:, anchor_index : anchor_index + 1, 3:7].repeat(1, num_bodies, 1),
        robot_state.body_state[:, body_indexes, :3],
        robot_state.body_state[:, body_indexes, 3:7],
    )  # [n_envs, n_bodies, 3] positions of each body relative to the anchor frame

    return pos_b.view(env.num_envs, -1)  # [n_envs, n_bodies * 3]


def robot_body_ori_b(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Body orientations relative to anchor frame."""
    robot_state = env_states.robots[env.name]
    body_indexes = env.commands.body_indexes
    anchor_index = env.commands.robot_anchor_body_index

    num_bodies = len(env.commands.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        robot_state.body_state[:, anchor_index : anchor_index + 1, :3].repeat(1, num_bodies, 1),
        robot_state.body_state[:, anchor_index : anchor_index + 1, 3:7].repeat(1, num_bodies, 1),
        robot_state.body_state[:, body_indexes, :3],
        robot_state.body_state[:, body_indexes, 3:7],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Target anchor position relative to anchor frame."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index

    pos, _ = subtract_frame_transforms(
        robot_state.body_state[:, anchor_index, :3],
        robot_state.body_state[:, anchor_index, 3:7],
        env.commands.anchor_pos_w,
        env.commands.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Target anchor orientation relative to anchor frame."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index

    _, ori = subtract_frame_transforms(
        robot_state.body_state[:, anchor_index, :3],  # [n_envs, 3]
        robot_state.body_state[:, anchor_index, 3:7],  # [n_envs, 4]
        env.commands.anchor_pos_w,  # [n_envs, 3]
        env.commands.anchor_quat_w,  # [n_envs, 4]
    )  # [n_envs, 4] quaternion representing the relative rotation between the two frames
    mat = matrix_from_quat(ori)  # [n_envs, 3, 3] convert to rotation matrix
    return mat[..., :2].reshape(
        mat.shape[0], -1
    )  # [n_envs, 6] extract the first two rows because the third row can be derived from orthogonality


def generated_commands(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # return env.command_manager.get_command(command_name)
    return env.commands.command


def base_lin_vel(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Root linear velocity in the robot's root frame."""
    robot_state = env_states.robots[env.name]
    return quat_rotate_inverse(robot_state.root_state[:, 3:7], robot_state.root_state[:, 7:10])


def base_ang_vel(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Root angular velocity in the robot's root frame."""
    robot_state = env_states.robots[env.name]
    return quat_rotate_inverse(robot_state.root_state[:, 3:7], robot_state.root_state[:, 10:13])


def joint_pos_rel(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """The joint positions of the robot w.r.t. the default joint positions."""
    robot_state = env_states.robots[env.name]
    joint_pos_sorted = robot_state.joint_pos - env.default_dof_pos_sorted
    return joint_pos_sorted[:, env.sorted_to_original_joint_indexes]


def joint_vel_rel(env: LeggedRobotTask, env_states: TensorState):
    """The joint velocities of the robot w.r.t. the default joint velocities."""
    robot_state = env_states.robots[env.name]
    joint_vel_sorted = robot_state.joint_vel - env.default_dof_vel_sorted
    return joint_vel_sorted[:, env.sorted_to_original_joint_indexes]  # (n_envs, n_dofs)


def last_action(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """The last input action to the environment."""
    return env._action
