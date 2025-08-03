"""This file contains the basic types for the MetaSim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

import torch

from metasim.utils.math import convert_camera_frame_orientation_convention

## Basic types
Dof = Dict[str, float]


## Trajectory types
class RobotAction(TypedDict):
    """Action of the robot."""

    dof_pos_target: Dof
    dof_effort_target: Dof | None


Action = Dict[str, RobotAction]


class ObjectState(TypedDict):
    """State of the object."""

    pos: torch.Tensor
    rot: torch.Tensor
    vel: torch.Tensor | None
    ang_vel: torch.Tensor | None
    dof_pos: Dof | None
    dof_vel: Dof | None
    com: torch.Tensor | None
    com_vel: torch.Tensor | None


class RobotState(ObjectState):
    """State of the robot."""

    dof_pos: Dof | None
    dof_vel: Dof | None
    dof_torque: Dof | None


class EnvState(TypedDict):
    """State of the environment."""

    objects: dict[str, ObjectState]
    robots: dict[str, RobotState]
    cameras: dict[str, dict[str, torch.Tensor]]


@dataclass
class ContactForceState:
    """State of a single contact force sensor."""

    force: torch.Tensor
    """Contact force. Shape is (num_envs, 3)."""


SensorState = ContactForceState


@dataclass
class ObjectState:
    """State of a single object."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str] | None = None
    """Body names. This is only available for articulation objects."""
    body_state: torch.Tensor | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
    joint_pos: torch.Tensor | None = None
    """Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
    joint_vel: torch.Tensor | None = None
    """Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""


@dataclass
class RobotState:
    """State of a single robot."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str]
    """Body names."""
    body_state: torch.Tensor
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
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

    ## Images
    rgb: torch.Tensor | None
    """RGB image. Shape is (num_envs, H, W, 3)."""
    depth: torch.Tensor | None
    """Depth image. Shape is (num_envs, H, W)."""
    instance_id_seg: torch.Tensor | None = None
    """Instance id segmentation for each pixel. Shape is (num_envs, H, W)."""
    instance_id_seg_id2label: dict[int, str] | None = None
    """Instance id segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_id_seg`."""
    instance_seg: torch.Tensor | None = None
    """Instance segmentation for each pixel. Shape is (num_envs, H, W).

    .. warning::
        This is experimental and subject to change.
    """
    instance_seg_id2label: dict[int, str] | None = None
    """Instance segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_seg`.

    .. warning::
        This is experimental and subject to change.
    """

    ## Camera parameters
    pos: torch.Tensor | None = None  # TODO: remove N
    """Position of the camera. Shape is (num_envs, 3)."""
    quat_world: torch.Tensor | None = None  # TODO: remove N
    """Quaternion ``(w, x, y, z)`` of the camera, following the world frame convention. Shape is (num_envs, 4).

    Note:
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.
    """
    intrinsics: torch.Tensor | None = None  # TODO: remove N
    """Intrinsics matrix of the camera. Shape is (num_envs, 3, 3)."""

    @property
    def quat_ros(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the ROS convention. Shape is (num_envs, 4).

        Note:
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="ros")

    @property
    def quat_opengl(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the OpenGL convention. Shape is (num_envs, 4).

        Note:
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="opengl")


@dataclass
class TensorState:
    """Tensorized state of the simulation."""

    objects: dict[str, ObjectState]
    """States of all objects."""
    robots: dict[str, RobotState]
    """States of all robots."""
    cameras: dict[str, CameraState]
    """States of all cameras."""
    sensors: dict[str, SensorState]
    """States of all sensors."""
    extras: dict = field(default_factory=dict)
    """States of Extra information"""
