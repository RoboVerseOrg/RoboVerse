"""This module provides utility functions for kinematics calculations using both curobo and pyroki libraries."""

import torch

from metasim.scenario.robot import RobotCfg
from metasim.utils.math import matrix_from_quat


def get_curobo_models(robot_cfg: RobotCfg, no_gnd=False):
    """Initializes and returns the curobo kinematic model, forward kinematics function, and inverse kinematics solver for a given robot configuration.

    Args:
        robot_cfg (RobotCfg): The configuration object for the robot.
        no_gnd (bool, optional): Whether to exclude ground collision objects. Defaults to False.

    Returns:
        tuple: A tuple containing the curobo kinematic model, forward kinematics function, and IK solver.
    """
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_path, join_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

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


def get_pyroki_model(robot_cfg: RobotCfg):
    """Get the Pyroki robot model and IK solver.

    Args:
        robot_cfg: Robot configuration containing urdf_path and ee_body_name.

    Returns:
        dict: Dictionary containing the pyroki robot model and solve_ik function.
    """
    import pyroki as pk
    from yourdfpy import URDF

    from metasim.utils.hf_util import check_and_download_single

    urdf_path = robot_cfg.urdf_path
    check_and_download_single(urdf_path)
    ee_link_name = getattr(robot_cfg, "ee_body_name", None)
    if ee_link_name is None:
        raise ValueError("robot_cfg must have 'ee_body_name' defined")

    # Load URDF model from file
    urdf = URDF.load(urdf_path, load_meshes=False)

    # Initialize Pyroki robot model from URDF
    pk_robot = pk.Robot.from_urdf(urdf)

    import jax_dataclasses as jdc
    import jaxlie
    import jaxls

    @jdc.jit
    def _solve_ik_jax_with_seed(robot, target_link_index, target_wxyz, target_position, prev_cfg):
        joint_var = robot.joint_var_cls(0)
        factors = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
                target_link_index,
                pos_weight=50.0,
                ori_weight=10.0,
            ),
            pk.costs.limit_cost(robot, joint_var, weight=100.0),
        ]

        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.00),
            )
        )
        return sol[joint_var]

    @jdc.jit
    def _solve_ik_jax_without_seed(robot, target_link_index, target_wxyz, target_position):
        joint_var = robot.joint_var_cls(0)
        factors = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
                target_link_index,
                pos_weight=50.0,
                ori_weight=10.0,
            ),
            pk.costs.limit_cost(robot, joint_var, weight=100.0),
        ]

        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.00),
            )
        )
        return sol[joint_var]

    def solve_ik_with_seed(pos_target: torch.Tensor, quat_target: torch.Tensor, seed_q: torch.Tensor) -> torch.Tensor:
        import jax.numpy as jnp
        import numpy as np

        pos_np = pos_target.detach().cpu().numpy()
        quat_np = quat_target.detach().cpu().numpy()
        seed_np = seed_q.detach().cpu().numpy()
        target_link_index = pk_robot.links.names.index(ee_link_name)

        num_actuated_joints = pk_robot.joints.num_actuated_joints
        if seed_np.shape[0] != num_actuated_joints:
            if seed_np.shape[0] < num_actuated_joints:
                padding = np.zeros(num_actuated_joints - seed_np.shape[0])
                seed_np = np.concatenate([seed_np, padding])
            else:
                seed_np = seed_np[:num_actuated_joints]

        cfg = _solve_ik_jax_with_seed(
            pk_robot, jnp.array(target_link_index), jnp.array(quat_np), jnp.array(pos_np), jnp.array(seed_np)
        )
        return torch.from_numpy(np.array(cfg)).to(pos_target.device)

    def solve_ik_without_seed(pos_target: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
        import jax.numpy as jnp
        import numpy as np

        pos_np = pos_target.detach().cpu().numpy()
        quat_np = quat_target.detach().cpu().numpy()
        target_link_index = pk_robot.links.names.index(ee_link_name)

        cfg = _solve_ik_jax_without_seed(pk_robot, jnp.array(target_link_index), jnp.array(quat_np), jnp.array(pos_np))
        return torch.from_numpy(np.array(cfg)).to(pos_target.device)

    return {
        "robot": pk_robot,
        "solve_ik_with_seed": solve_ik_with_seed,
        "solve_ik_without_seed": solve_ik_without_seed,
        "ee_link_name": ee_link_name,
    }


# ==================== TCP/EE POSE CONVERSION FUNCTIONS ====================


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


# ==================== UTILITY FUNCTIONS ====================


def _quat_wxyz_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> Euler RPY (radians), intrinsic XYZ."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)

    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))

    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


# ==================== END-EFFECTOR STATE FUNCTIONS ====================


def get_ee_state(obs, robot_config, tensorize: bool = False, use_rpy: bool = True):
    """Return EE state.

    - if use_rpy=True: [pos(3), rpy(3), grip(1)]
    - else:            [pos(3), quat(4), grip(1)]
    Grip is binary (1=open, 0=close) from finger DOFs.
    """
    rs = obs.robots[robot_config.name]
    device = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).device

    body_state = (
        rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state, device=device).float()
    )
    joint_pos = (
        rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos, device=device).float()
    )

    ee_body_index = rs.body_names.index(robot_config.ee_body_name)
    ee_joint_indices = robot_config.ee_joint_indices

    ee_pos_world = body_state[:, ee_body_index, 0:3]  # (B,3)
    ee_quat_world = body_state[:, ee_body_index, 3:7]  # (B,4) wxyz

    # gripper open/close
    joint_pos_gripper = (
        joint_pos[:, ee_joint_indices] if len(ee_joint_indices) > 1 else joint_pos[:, ee_joint_indices[0]].unsqueeze(-1)
    )
    open_q = torch.as_tensor(robot_config.gripper_open_q, device=device, dtype=joint_pos_gripper.dtype).view(1, -1)
    close_q = torch.as_tensor(robot_config.gripper_close_q, device=device, dtype=joint_pos_gripper.dtype).view(1, -1)
    denom = (open_q - close_q).clamp_min(1e-6)
    open_ratio = ((joint_pos_gripper - close_q) / denom).clamp(0.0, 1.0)
    gripper_open_binary = (open_ratio.mean(dim=-1) >= 0.5).to(joint_pos_gripper.dtype).unsqueeze(-1)  # (B,1)

    if use_rpy:
        ee_rpy_world = _quat_wxyz_to_rpy(ee_quat_world)  # (B,3)
        ee_flat_world = torch.cat([ee_pos_world, ee_rpy_world, gripper_open_binary], dim=-1)  # (B,7)
    else:
        ee_flat_world = torch.cat([ee_pos_world, ee_quat_world, gripper_open_binary], dim=-1)  # (B,8)

    return ee_flat_world if tensorize else [{"ee_state": ee_flat_world[i]} for i in range(ee_flat_world.shape[0])]


def get_ee_state_from_list(env_states, robot_config, tensorize: bool = False, use_rpy: bool = True):
    """Return per-env EE state.

    - if use_rpy=True: [pos(3), rpy(3), grip(1)]
    - else:            [pos(3), quat(4), grip(1)]
    """
    robot_name = robot_config.name
    ee_body_name = robot_config.ee_body_name
    ee_joint_names = robot_config.ee_joint_names

    gripper_open_q = torch.as_tensor(robot_config.gripper_open_q, dtype=torch.float32).view(-1)
    gripper_close_q = torch.as_tensor(robot_config.gripper_close_q, dtype=torch.float32).view(-1)
    if gripper_open_q.numel() != len(ee_joint_names) or gripper_close_q.numel() != len(ee_joint_names):
        raise ValueError("gripper_(open|close)_q must match ee_joint_names length")

    ee_states = []
    for env_state in env_states:
        robot_state = env_state["robots"][robot_name]

        # Skip if "body" field is not present in the state
        if "body" not in robot_state:
            continue

        body_state = robot_state["body"][ee_body_name]

        ee_pos_world = torch.as_tensor(body_state["pos"], dtype=torch.float32).view(3)  # (3,)
        ee_quat_world = torch.as_tensor(body_state["rot"], dtype=torch.float32).view(4)  # (4,) wxyz

        # finger DOFs
        finger_positions = torch.stack(
            [torch.as_tensor(robot_state["dof_pos"][j], dtype=torch.float32) for j in ee_joint_names], dim=0
        )  # (n_fingers,)
        denom = (gripper_open_q - gripper_close_q).clamp_min(1e-6)
        open_ratio = ((finger_positions - gripper_close_q) / denom).clamp(0.0, 1.0)
        gripper_open_binary = (open_ratio.mean() >= 0.5).to(torch.float32).view(1)  # (1,)

        if use_rpy:
            ee_rpy_world = _quat_wxyz_to_rpy(ee_quat_world.view(1, 4)).view(3)  # (3,)
            ee_vec = torch.cat([ee_pos_world, ee_rpy_world, gripper_open_binary], dim=0)  # (7,)
        else:
            ee_vec = torch.cat([ee_pos_world, ee_quat_world, gripper_open_binary], dim=0)  # (8,)

        ee_states.append(ee_vec)

    ee_states = torch.stack(ee_states, dim=0)
    return ee_states if tensorize else [{"ee_state": ee_states[i]} for i in range(ee_states.shape[0])]
