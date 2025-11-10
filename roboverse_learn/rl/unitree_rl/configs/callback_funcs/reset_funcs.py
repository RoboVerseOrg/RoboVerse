from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_from_euler_xyz, sample_uniform, quat_mul

from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes


def random_root_state(env: EnvTypes, env_ids: torch.Tensor | list, pose_range: list[list]=[[0]*6, [0]*6], velocity_range: list[list]=[[0]*6, [0]*6]) -> torch.Tensor:
    if len(env_ids) == 0:
        return

    root_states = env.default_env_states.robots[env.name].root_state[env_ids].clone()

    # poses
    pose_range = torch.tensor(pose_range, device=env.device)
    rand_samples = sample_uniform(pose_range[0], pose_range[1], (len(env_ids), 6), device=env.device)
    positions = root_states[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    velocity_range = torch.tensor(velocity_range, device=env.device)
    rand_samples = sample_uniform(velocity_range[0], velocity_range[1], (len(env_ids), 6), device=env.device)

    velocities = root_states[:, 7:13] + rand_samples

    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 0:3] = positions
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 3:7] = orientations
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 7:13] = velocities
    # # set into the physics simulation
    # env.write_robot_root_state(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)


def reset_joints_by_scale(env: EnvTypes, env_ids: torch.Tensor | list, position_range: list|tuple=(1.0, 1.0), velocity_range: list|tuple=(1.0, 1.0)) -> torch.Tensor:
    if len(env_ids) == 0:
        return

    # get default joint state
    joint_pos = env.default_env_states.robots[env.name].joint_pos[env_ids].clone()
    joint_vel = env.default_env_states.robots[env.name].joint_vel[env_ids].clone()

    # scale these values randomly
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = env.soft_dof_pos_limits
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = env.soft_dof_vel_limits
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    env.setup_initial_env_states.robots[env.name].joint_pos[env_ids] = joint_pos
    env.setup_initial_env_states.robots[env.name].joint_vel[env_ids] = joint_vel
    # # set into the physics simulation
    # asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
