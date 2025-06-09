"""This file contains the basic types for the MetaSim."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import torch

## Basic types
Dof = Dict[str, float]


## Trajectory types
class RobotAction(TypedDict):
    """Action of the robot."""

    dof_pos_target: Dof


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

    dof_pos_target: Dof | None
    dof_vel_target: Dof | None
    dof_torque: Dof | None


class EnvState(TypedDict):
    """State of the environment."""

    objects: dict[str, ObjectState]
    robots: dict[str, RobotState]
    cameras: dict[str, dict[str, torch.Tensor]]


## Gymnasium types
Obs = List[EnvState]
Reward = List[List[float]]  # TODO: you may modify this if necessary
Success = torch.BoolTensor
TimeOut = torch.BoolTensor
Extra = Dict[str, Any]  # XXX
Termination = torch.BoolTensor
