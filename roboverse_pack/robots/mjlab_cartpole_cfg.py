"""Minimal RobotCfg wrapper around mjlab's cartpole.xml.

Lets cartpole_v2 task work on backends (Newton, MJX) that require a
formal RobotCfg — currently the task uses scene-MJCF which only works
on MujocoHandler.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class MjlabCartpoleCfg(RobotCfg):
    """Cartpole from mjlab/tasks/cartpole/cartpole.xml (2 hinge joints, 1 slide actuator)."""

    # NOTE: keep name == "cartpole" to align with the SceneEntityCfg names
    # in roboverse_pack/tasks/mjlab/cartpole_v2.py — that way the same task
    # works on both backends (mujoco scene-MJCF path + newton RobotCfg path).
    name: str = "cartpole"
    mjcf_path: str = "roboverse_data/robots/mjlab/tasks/cartpole/cartpole.xml"
    urdf_path: str = MISSING  # MJCF only: the URDF and Isaac Sim backends report it as n/a
    usd_path: str = MISSING

    num_joints: int = 2
    fix_base_link: bool = True
    enabled_gravity: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "slider": BaseActuatorCfg(stiffness=0, damping=0, effort_limit_sim=1.0, velocity_limit=20.0),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "slider": (-1.8, 1.8),  # the MJCF range (robot check limits row)
        "hinge_1": (-3.14159, 3.14159),
    }
    default_joint_positions: dict[str, float] = {
        "slider": 0.0,
        "hinge_1": 0.0,  # balance; swingup uses pi via reset event
    }
    default_joint_velocities: dict[str, float] = {
        "slider": 0.0,
        "hinge_1": 0.0,
    }
    control_type: dict[str, Literal["position", "effort"]] = {
        "slider": "effort",
    }
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
