from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from roboverse_pack.tasks.beyondmimic.isaaclab.mdp.rewards import _get_body_indexes

if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.isaaclab.envs.tracking_rl_env import TrackingRLEnv
    from roboverse_pack.tasks.beyondmimic.isaaclab.mdp.commands import MotionCommand


def bad_anchor_pos(env: TrackingRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Distance between target and actual anchor position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


# `anchor_pos_w` is of shape [n_envs, 3] and -1 retrieves the Z coordinate
def bad_anchor_pos_z_only(env: TrackingRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Distance between target and actual anchor position in the Z direction."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(env: TrackingRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """Distance between target and actual anchor orientation."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    # converts world-frame gravity vector to anchor frame
    motion_projected_gravity_b = math_utils.quat_apply_inverse(
        command.anchor_quat_w, asset.data.GRAVITY_VEC_W
    )  # [n_envs, 3]

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    # checks whether the robot’s tilt magnitude deviates too much (how relatively "upright"), and ignores which way it leans
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: TrackingRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Distance between target and actual body position."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: TrackingRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Distance between target and actual body position in the Z direction."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
