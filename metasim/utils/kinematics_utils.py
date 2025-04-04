"""This module provides utility functions for kinematics calculations using the curobo library."""

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.utils.math import matrix_from_quat


def get_curobo_models(robot_cfg: BaseRobotCfg, no_gnd=False):
    """Initializes and returns the curobo kinematic model, forward kinematics function, and inverse kinematics solver for a given robot configuration.

    Args:
        robot_cfg (BaseRobotCfg): The configuration object for the robot.
        no_gnd (bool, optional): If True, the ground plane is not included for curobo collision checking. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - kin_model (CudaRobotModel): The kinematic model of the robot.
            - do_fk (function): A function that performs forward kinematics given joint positions.
            - ik_solver (IKSolver): The inverse kinematics solver configured for the robot.
    """
    tensor_args = TensorDeviceType()
    config_file = load_yaml(join_path(get_robot_path(), robot_cfg.curobo_ref_cfg_name))["robot_cfg"]
    curobo_robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="ground",
                pose=[0.0, 0.0, -0.4, 1, 0.0, 0.0, 0.0],
                dims=[10.0, 10.0, 0.8],
            )
        ]
    )
    ik_config = IKSolverConfig.load_from_robot_config(
        curobo_robot_cfg,
        None if no_gnd else world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )

    ik_solver = IKSolver(ik_config)
    kin_model = CudaRobotModel(curobo_robot_cfg.kinematics)

    def do_fk(q: torch.Tensor):
        robot_state = kin_model.get_state(q, config_file["kinematics"]["ee_link"])
        return robot_state.ee_position, robot_state.ee_quaternion

    return kin_model, do_fk, ik_solver


def ee_pose_from_tcp_pose(robot_cfg: BaseRobotCfg, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor):
    """Calculate the end-effector (EE) pose from the tool center point (TCP) pose.

    Note that currently only the translation is considered.

    Args:
        robot_cfg (BaseRobotCfg): Configuration object for the robot, containing the relative position of the TCP.
        tcp_pos (torch.Tensor): The position of the TCP as a tensor.
        tcp_quat (torch.Tensor): The orientation of the TCP as a tensor, in scalar-first quaternion.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The position and orientation of the end-effector.
    """
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(tcp_pos.device)
    ee_pos = tcp_pos + torch.matmul(matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)).squeeze()
    return ee_pos, tcp_quat


def tcp_pose_from_ee_pose(robot_cfg: BaseRobotCfg, ee_pos: torch.Tensor, ee_quat: torch.Tensor):
    """Calculate the TCP (Tool Center Point) pose from the end-effector pose.

    Note that currently only the translation is considered.

    Args:
        robot_cfg (BaseRobotCfg): Configuration object for the robot, containing the relative position of the TCP.
        ee_pos (torch.Tensor): The position of the end-effector as a tensor.
        ee_quat (torch.Tensor): The orientation of the end-effector as a tensor, in scalar-first quaternion.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The position and orientation of the end-effector.
    """
    ee_rotmat = matrix_from_quat(ee_quat)
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(ee_rotmat.device)
    tcp_pos = ee_pos + torch.matmul(ee_rotmat, tcp_rel_pos.unsqueeze(-1)).squeeze()
    return tcp_pos, ee_quat
