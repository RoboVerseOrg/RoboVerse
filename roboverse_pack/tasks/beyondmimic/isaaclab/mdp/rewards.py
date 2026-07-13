# Copyright (c) The BeyondMimic authors (HybridRobotics/whole_body_tracking)
# SPDX-License-Identifier: MIT
#
# Adapted from BeyondMimic / whole_body_tracking (https://github.com/HybridRobotics/whole_body_tracking).
# Changes: import paths rewired to RoboVerse; env type hints retargeted to TrackingRLEnv; docstrings added.
# Full license: roboverse_pack/tasks/beyondmimic/LICENSE.beyondmimic

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.isaaclab.envs.tracking_rl_env import TrackingRLEnv
    from roboverse_pack.tasks.beyondmimic.isaaclab.mdp.commands import MotionCommand


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: TrackingRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Distance between target and actual anchor position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: TrackingRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Distance between target and actual anchor orientation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: TrackingRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Distance between target and actual body position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: TrackingRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Distance between target and actual body orientation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: TrackingRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Linear velocity tracking error."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: TrackingRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Distance between target and actual body linear velocity."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: TrackingRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Time spent in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
