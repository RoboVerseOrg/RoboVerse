from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse

from roboverse_learn.rl.unitree_rl.configs.cfg_queries import ContactForces
from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes
from roboverse_learn.rl.unitree_rl.helper import get_indices_from_substring, hash_names


def track_lin_vel_xy(
    env: EnvTypes, env_states: TensorState, std: float
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    lin_vel_diff = env.commands_manager.value[:, :2] - base_lin_vel[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z(env: EnvTypes, env_states: TensorState, std: float) -> torch.Tensor:
    """
    Track angular velocity commands (yaw).
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
    ang_vel_diff = env.commands_manager.value[:, 2] - base_ang_vel[:, 2]
    ang_vel_error = torch.square(ang_vel_diff)
    return torch.exp(-ang_vel_error / std**2)


def is_alive(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.reset_buf).float()
    # return 1.0


def lin_vel_z(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    return torch.square(base_lin_vel[:, 2])


def ang_vel_xy(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)


def joint_vel(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    return torch.sum(torch.square(robot_state.joint_vel), dim=1)


def joint_acc(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    return torch.sum(
        torch.square(
            (env.history_buffer["joint_vel"][-1] - robot_state.joint_vel) / env.step_dt
        ),
        dim=1,
    )


def action_rate(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.history_buffer["actions"][-1] - env.actions), dim=1
    )


def joint_pos_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    robot_state = env_states.robots[env.name]
    out_of_limits = -(robot_state.joint_pos - env.soft_dof_pos_limits[:, 0]).clip(
        max=0.0
    )
    out_of_limits += (robot_state.joint_pos - env.soft_dof_pos_limits[:, 1]).clip(
        min=0.0
    )
    return torch.sum(out_of_limits, dim=1)


def energy(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Sum |qdot|*|tau| across joints (\"energy\" usage)."""
    base = env_states.robots[env.name]
    qvel = base.joint_vel
    qfrc = base.joint_effort_target
    # qfrc = env.torques # TODO: wait isaacsim handler complete dof_torques in robot_state
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def _get_indices(
    env: EnvTypes, sub_names: tuple[str] | str, all_names: list[str] | tuple[str]
):
    hash_key = hash_names(sub_names)
    if hash_key not in env.extras_buffer:
        env.extras_buffer[hash_key] = get_indices_from_substring(
            sub_names, all_names, fullmatch=True
        ).to(env.device)
    return env.extras_buffer[hash_key]


def joint_deviation_l1(
    env: EnvTypes, env_states: TensorState, joint_names: str | tuple[str]
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    indices = _get_indices(env, joint_names, env.sorted_joint_names)
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    # compute out of limits constraints
    angle = robot_state.joint_pos[:, indices] - env.default_dof_pos[indices]
    return torch.sum(torch.abs(angle), dim=1)


def flat_orientation(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def base_height(
    env: EnvTypes, env_states: TensorState, target_height: float
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_height = robot_state.root_state[:, 2]
    if False:  # height scanner
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(
            sensor.data.ray_hits_w[..., 2], dim=1
        )
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(base_height - adjusted_target_height)


def feet_gait(
    env: EnvTypes,
    env_states: TensorState,
    period: float,
    offset: list[float],
    threshold: float = 0.55,
    body_names: str | tuple[str] = ".*ankle_roll.*",
) -> torch.Tensor:
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    command_name = "base_velocity"

    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = (
        contact_forces.contact_forces_history[:, :, indices, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # contact_sensor = env.handler.contact_sensor
    # is_contact = contact_sensor.data.current_contact_time[:, env.body_ids_reindex][:, env.extras_buffer[bodies_key]] > 0

    # # #### Implemention 2: using sine wave phase
    # global_phase = (env._episode_steps * env.step_dt) % period / period
    # sin_pos = torch.sin(2 * torch.pi * global_phase)
    # # Add double support phase
    # is_stance = torch.zeros(
    #     (env.num_envs, len(indices)), dtype=torch.bool, device=env.device
    # )
    # # left foot stance
    # is_stance[:, 0] = sin_pos >= 0
    # # right foot stance
    # is_stance[:, 1] = sin_pos < 0
    # # Double support phase
    # is_stance[torch.abs(sin_pos) < threshold - 0.5] = True

    # reward = torch.sum(is_contact == is_stance, dim=1, dtype=torch.float32)

    #### Implemention 1: using phase offsets
    global_phase = ((env._episode_steps * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(indices)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name == "base_velocity":
        cmd_norm = torch.norm(env.commands_manager.value[:, :2], dim=1)
        reward *= (cmd_norm > 0.1).float()
    return reward


def feet_slide(
    env: EnvTypes,
    env_states: TensorState,
    body_names: str | tuple[str] = ".*ankle_roll.*",
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)

    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    contacts = (
        contact_forces.contact_forces_history[:, :, indices, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    body_vel = env_states.robots[env.name].body_state[:, indices, 7:9]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_clearance(
    env: EnvTypes,
    env_states: TensorState,
    target_height: float,
    std: float,
    tanh_mult: float,
    body_names: str | tuple[str] = ".*ankle_roll.*",
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    base = env_states.robots[env.name]
    foot_z_target_error = torch.square(base.body_state[:, indices, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(base.body_state[:, indices, 7:9], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std**2)


def undesired_contacts(
    env: EnvTypes,
    env_states: TensorState,
    threshold: float,
    body_names: str | tuple[str] = "(?!.*ankle.*).*",
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = (
        contact_forces.contact_forces_history[:, :, indices, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )

    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)
