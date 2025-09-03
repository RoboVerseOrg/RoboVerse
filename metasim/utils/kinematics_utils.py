"""This module provides utility functions for kinematics calculations using the curobo library."""

import torch

from metasim.scenario.robot import RobotCfg
from metasim.utils.math import matrix_from_quat


def get_curobo_models(robot_cfg: RobotCfg, no_gnd=False):
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_path, join_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    """Initializes and returns the curobo kinematic model, forward kinematics function, and inverse kinematics solver for a given robot configuration.

    Args:
        robot_cfg (RobotCfg): The configuration object for the robot.
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


def ee_pose_from_tcp_pose(robot_cfg: RobotCfg, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor):
    """Calculate the end-effector (EE) pose from the tool center point (TCP) pose.

    Note that currently only the translation is considered.

    Args:
        robot_cfg (RobotCfg): Configuration object for the robot, containing the relative position of the TCP.
        tcp_pos (torch.Tensor): The position of the TCP as a tensor.
        tcp_quat (torch.Tensor): The orientation of the TCP as a tensor, in scalar-first quaternion.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The position and orientation of the end-effector.
    """
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(tcp_pos.device)
    ee_pos = tcp_pos + torch.matmul(matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)).squeeze()
    return ee_pos, tcp_quat


def tcp_pose_from_ee_pose(robot_cfg: RobotCfg, ee_pos: torch.Tensor, ee_quat: torch.Tensor):
    """Calculate the TCP (Tool Center Point) pose from the end-effector pose.

    Note that currently only the translation is considered.

    Args:
        robot_cfg (RobotCfg): Configuration object for the robot, containing the relative position of the TCP.
        ee_pos (torch.Tensor): The position of the end-effector as a tensor.
        ee_quat (torch.Tensor): The orientation of the end-effector as a tensor, in scalar-first quaternion.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The position and orientation of the end-effector.
    """
    ee_rotmat = matrix_from_quat(ee_quat)
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(ee_rotmat.device)
    tcp_pos = ee_pos + torch.matmul(ee_rotmat, tcp_rel_pos.unsqueeze(-1)).squeeze()
    return tcp_pos, ee_quat



def get_ee_state(obs, robot_config, tensorize=False):
    rs = obs.robots[robot_config.name]
    device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor)
              else torch.tensor(rs.joint_pos)).device

    body_state = (rs.body_state if isinstance(rs.body_state, torch.Tensor)
                  else torch.tensor(rs.body_state, device=device).float())
    joint_pos = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor)
                 else torch.tensor(rs.joint_pos, device=device).float())

    ee_idx = rs.body_names.index(robot_config.ee_body_name)
    ee_joint_idx = rs.joint_names.index(robot_config.ee_joint_name)

    ee_pos_world  = body_state[:, ee_idx, 0:3]   # (B,3)
    ee_quat_world = body_state[:, ee_idx, 3:7]   # (B,4)

    joint_pos_grip = joint_pos[:, ee_joint_idx]  # (B,K)
    open_q  = torch.as_tensor(robot_config.gripper_open_q,  device=device,
                              dtype=joint_pos_grip.dtype).view(1, -1)  # (1,K)
    close_q = torch.as_tensor(robot_config.gripper_close_q, device=device,
                              dtype=joint_pos_grip.dtype).view(1, -1)  # (1,K)

    denom = open_q - close_q
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    open_per_finger = ((joint_pos_grip - close_q) / denom).clamp(0, 1)  # (B,K)
    gripper_open = open_per_finger.mean(dim=-1)                          # (B,)

    ee_flat_world = torch.cat([ee_pos_world, ee_quat_world, gripper_open.unsqueeze(-1)], dim=-1)  # (B,8)

    if tensorize:
        return ee_flat_world
    else:
        return [{"ee_state": ee_flat_world[i]} for i in range(ee_flat_world.shape[0])]
