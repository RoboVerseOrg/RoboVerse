from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_error_magnitude
from roboverse_pack.tasks.beyondmimic.metasim.configs.cfg_queries import ContactForces
from roboverse_pack.tasks.beyondmimic.metasim.utils.misc import get_body_indexes
from roboverse_pack.tasks.beyondmimic.metasim.utils.string import get_indexes_hash

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.metasim.envs.base_legged_robot import LeggedRobotTask


def motion_global_anchor_position_error_exp(env: LeggedRobotTask, env_states: TensorState, std: float) -> torch.Tensor:
    """Global anchor position error."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index
    error = torch.sum(torch.square(env.commands.anchor_pos_w - robot_state.body_state[:, anchor_index, :3]), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: LeggedRobotTask, env_states: TensorState, std: float
) -> torch.Tensor:
    """Global anchor orientation error."""
    robot_state = env_states.robots[env.name]
    anchor_index = env.commands.robot_anchor_body_index
    error = quat_error_magnitude(env.commands.anchor_quat_w, robot_state.body_state[:, anchor_index, 3:7]) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: LeggedRobotTask, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Relative body position error."""
    body_state = env_states.robots[env.name].body_state[:, env.commands.body_indexes, :]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = torch.sum(
        torch.square(env.commands.body_pos_relative_w[:, body_indexes] - body_state[:, body_indexes, :3]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: LeggedRobotTask, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Relative body orientation error."""
    body_state = env_states.robots[env.name].body_state[:, env.commands.body_indexes, :]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = (
        quat_error_magnitude(env.commands.body_quat_relative_w[:, body_indexes], body_state[:, body_indexes, 3:7]) ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: LeggedRobotTask, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Linear velocity tracking error."""
    body_state = env_states.robots[env.name].body_state[:, env.commands.body_indexes, :]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = torch.sum(
        torch.square(env.commands.body_lin_vel_w[:, body_indexes] - body_state[:, body_indexes, 7:10]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: LeggedRobotTask, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Angular velocity tracking error."""
    body_state = env_states.robots[env.name].body_state[:, env.commands.body_indexes, :]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = torch.sum(
        torch.square(env.commands.body_ang_vel_w[:, body_indexes] - body_state[:, body_indexes, 10:13]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def action_rate_l2(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    # NOTE `env.actions` is already in the original order
    return torch.sum(torch.square(env._action - env._prev_action), dim=1)  # [n_envs, n_dims]


def joint_pos_limits(env: LeggedRobotTask, env_states: TensorState) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    robot_state = env_states.robots[env.name]
    out_of_limits = -(robot_state.joint_pos - env.soft_dof_pos_limits_sorted[:, :, 0]).clip(max=0.0)
    out_of_limits += (robot_state.joint_pos - env.soft_dof_pos_limits_sorted[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def undesired_contacts(
    env: LeggedRobotTask,
    env_states: TensorState,
    threshold: float,
    body_names: str | tuple[str] = "(?!.*ankle.*).*",
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # body_indexes = get_body_indexes(env.commands, body_names)
    indexes = get_indexes_hash(env, body_names, env_states.robots[env.name].body_names)
    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = (
        # TODO check correspondence with `contact_sensor.data.net_forces_w_history`
        contact_forces.contact_forces_history[
            :, :, indexes, :
        ]  # [n_envs, history_length, n_indexes, 3] -> [n_envs, 3, 26, 3]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )

    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)  # [n_envs]
