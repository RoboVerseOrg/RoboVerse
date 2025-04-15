"""Tensorized state of the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from loguru import logger as log

from metasim.types import EnvState

try:
    from metasim.sim.base import BaseSimHandler
except:
    pass


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


def _dof_tensor_to_dict(dof_tensor: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    """Convert a DOF tensor to a dictionary of joint positions."""
    joint_names = sorted(joint_names)
    return {jn: dof_tensor[i].item() for i, jn in enumerate(joint_names)}


def _body_tensor_to_dict(body_tensor: torch.Tensor, body_names: list[str]) -> dict[str, float]:
    """Convert a body tensor to a dictionary of body positions."""
    body_names = sorted(body_names)
    return {
        bn: {
            "pos": body_tensor[i][:3].cpu(),
            "rot": body_tensor[i][3:7].cpu(),
            "vel": body_tensor[i][7:10].cpu(),
            "ang_vel": body_tensor[i][10:13].cpu(),
        }
        for i, bn in enumerate(body_names)
    }


def state_tensor_to_nested(handler: BaseSimHandler, tensor_state: TensorState) -> list[EnvState]:
    """Convert a tensor state to a list of env states. All the tensors will be converted to cpu for compatibility."""
    log.warning(
        "Users please ignore this message, we are working on it. For developers: You are using the very inefficient function to convert the tensorized states to old nested states. Please consider not using this function and optimize your code when number of environments is large."
    )

    num_envs = next(iter(chain(tensor_state.objects.values(), tensor_state.robots.values()))).root_state.shape[0]
    env_states = []
    for env_id in range(num_envs):
        object_states = {}
        for obj_name, obj_state in tensor_state.objects.items():
            object_states[obj_name] = {
                "pos": obj_state.root_state[env_id, :3].cpu(),
                "rot": obj_state.root_state[env_id, 3:7].cpu(),
                "vel": obj_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": obj_state.root_state[env_id, 10:13].cpu(),
            }
            if obj_state.body_state is not None:
                bns = handler.get_body_names(obj_name)
                object_states[obj_name]["body"] = _body_tensor_to_dict(obj_state.body_state[env_id], bns)
            if obj_state.joint_pos is not None:
                jns = handler.get_object_joint_names(obj_name)
                object_states[obj_name]["dof_pos"] = _dof_tensor_to_dict(obj_state.joint_pos[env_id], jns)
            if obj_state.joint_vel is not None:
                jns = handler.get_object_joint_names(obj_name)
                object_states[obj_name]["dof_vel"] = _dof_tensor_to_dict(obj_state.joint_vel[env_id], jns)

        robot_states = {}
        for robot_name, robot_state in tensor_state.robots.items():
            jns = handler.get_object_joint_names(robot_name)
            robot_states[robot_name] = {
                "pos": robot_state.root_state[env_id, :3].cpu(),
                "rot": robot_state.root_state[env_id, 3:7].cpu(),
                "vel": robot_state.root_state[env_id, 7:10].cpu(),
                "ang_vel": robot_state.root_state[env_id, 10:13].cpu(),
            }
            robot_states[robot_name]["dof_pos"] = _dof_tensor_to_dict(robot_state.joint_pos[env_id], jns)
            robot_states[robot_name]["dof_vel"] = _dof_tensor_to_dict(robot_state.joint_vel[env_id], jns)
            robot_states[robot_name]["dof_pos_target"] = (
                _dof_tensor_to_dict(robot_state.joint_pos_target[env_id], jns)
                if robot_state.joint_pos_target is not None
                else None
            )
            robot_states[robot_name]["dof_vel_target"] = (
                _dof_tensor_to_dict(robot_state.joint_vel_target[env_id], jns)
                if robot_state.joint_vel_target is not None
                else None
            )
            robot_states[robot_name]["dof_torque"] = (
                _dof_tensor_to_dict(robot_state.joint_effort_target[env_id], jns)
                if robot_state.joint_effort_target is not None
                else None
            )
            if robot_state.body_state is not None:
                bns = handler.get_body_names(robot_name)
                robot_states[robot_name]["body"] = _body_tensor_to_dict(robot_state.body_state[env_id], bns)

        camera_states = {}
        for camera_name, camera_state in tensor_state.cameras.items():
            camera_states[camera_name] = {
                "rgb": camera_state.rgb[env_id].cpu(),
                "depth": camera_state.depth[env_id].cpu(),
            }

        env_state = {
            "objects": object_states,
            "robots": robot_states,
            "cameras": camera_states,
        }
        env_states.append(env_state)

    return env_states
