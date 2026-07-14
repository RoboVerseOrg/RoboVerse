# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: upstream's termination terms ported from Isaac Lab manager terms to MetaSim's TensorState-based callbacks.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_pack.tasks.beyondmimic.metasim.utils.misc import get_body_indexes

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.metasim.envs.base_legged_robot import LeggedRobotTask


# NOTE `env_states` is needed here for compatibility with how callbacks are triggered
def time_out(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env._episode_steps >= env.max_episode_steps


# NOTE bodies are in the original order


def bad_anchor_pos_z_only(env: LeggedRobotTask, env_states: TensorState, threshold: float) -> torch.Tensor:
    """Bad anchor position z-only."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index
    return torch.abs(env.commands.anchor_pos_w[:, 2] - robot_state.body_state[:, anchor_index, 2]) > threshold


def bad_anchor_ori(env: LeggedRobotTask, env_states: TensorState, threshold: float) -> torch.Tensor:
    """Bad anchor orientation."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index
    motion_projected_gravity_b = quat_rotate_inverse(env.commands.anchor_quat_w, env.gravity_vec)  # [n_envs, 3]
    robot_projected_gravity_b = quat_rotate_inverse(robot_state.body_state[:, anchor_index, 3:7], env.gravity_vec)

    # check whether the robot's tilt magnitude deviates too much (how relatively "upright"), and ignores which way it leans
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: LeggedRobotTask, env_states: TensorState, threshold: float, body_names: list[str]
) -> torch.Tensor:
    """Bad motion body position z-only."""
    body_state = env_states.robots[env.name].body_state[:, env.commands.body_indexes, :]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = torch.abs(env.commands.body_pos_relative_w[:, body_indexes, 2] - body_state[:, body_indexes, 2])
    return torch.any(error > threshold, dim=-1)
