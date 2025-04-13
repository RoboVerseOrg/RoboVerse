"""Tensorized state of the simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObjectState:
    """State of a single object."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    joint_pos: torch.Tensor | None = None
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: torch.Tensor | None = None
    """Joint velocities. Shape is (num_envs, num_joints)."""


@dataclass
class RobotState:
    """State of a single robot."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    joint_pos: torch.Tensor
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: torch.Tensor
    """Joint velocities. Shape is (num_envs, num_joints)."""
    joint_pos_target: torch.Tensor
    """Joint positions target. Shape is (num_envs, num_joints)."""
    joint_vel_target: torch.Tensor
    """Joint velocities target. Shape is (num_envs, num_joints)."""
    joint_effort_target: torch.Tensor
    """Joint effort targets. Shape is (num_envs, num_joints)."""


@dataclass
class CameraState:
    """State of a single camera."""

    rgb: torch.Tensor | None
    """RGB image. Shape is (num_envs, H, W, 3)."""
    depth: torch.Tensor | None
    """Depth image. Shape is (num_envs, H, W)."""


@dataclass
class TensorState:
    """Tensorized state of the simulation."""

    objects: dict[str, ObjectState]
    """States of all objects."""
    robots: dict[str, RobotState]
    """States of all robots."""
    cameras: dict[str, CameraState]
    """States of all cameras."""
