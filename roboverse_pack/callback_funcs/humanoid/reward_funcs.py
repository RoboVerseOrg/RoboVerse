from __future__ import annotations

import torch

from metasim.queries import ContactForces
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_pack.tasks.humanoid.base.types import EnvTypes
from roboverse_pack.utils.humanoid_utils import get_indices_from_substring, hash_names


def track_lin_vel_xy(env: EnvTypes, env_states: TensorState, std: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    lin_vel_diff = env.commands_manager.value[:, :2] - base_lin_vel[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z(env: EnvTypes, env_states: TensorState, std: float) -> torch.Tensor:
    """Track angular velocity commands (yaw)."""
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
        torch.square((env.history_buffer["joint_vel"][-1] - robot_state.joint_vel) / env.step_dt),
        dim=1,
    )


def action_rate(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.history_buffer["actions"][-1] - env.actions), dim=1)


def joint_pos_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    robot_state = env_states.robots[env.name]
    out_of_limits = -(robot_state.joint_pos - env.soft_dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (robot_state.joint_pos - env.soft_dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_effort_limits(env: EnvTypes, env_states: TensorState, soft_limit_factor: float = 1.0) -> torch.Tensor:
    """Penalize joint efforts that exceed torque limits using an L2 squared kernel."""
    robot_state = env_states.robots[env.name]
    if env.manual_pd_on:
        processed_actions = (env.actions * env.action_scale + env.actions_offset).clip(
            -env.action_clip,
            env.action_clip,
        )
        effort = env.p_gains * (processed_actions - robot_state.joint_pos) - env.d_gains * robot_state.joint_vel
    else:
        effort = robot_state.joint_effort_target
        if effort is None:
            return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    torque_limits = env.torque_limits * soft_limit_factor
    excess = torch.abs(effort) - torque_limits
    excess = excess.clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def energy(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    r"""Sum |qdot|*|tau| across joints ("energy" usage)."""
    base = env_states.robots[env.name]
    qvel = base.joint_vel
    qfrc = base.joint_effort_target
    # qfrc = env.torques # TODO: wait isaacsim handler complete dof_torques in robot_state
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def _get_indices(env: EnvTypes, sub_names: tuple[str] | str, all_names: list[str] | tuple[str]):
    hash_key = hash_names(sub_names)
    if hash_key not in env.extras_buffer:
        env.extras_buffer[hash_key] = get_indices_from_substring(sub_names, all_names, fullmatch=True).to(env.device)
    return env.extras_buffer[hash_key]


def joint_deviation_l1(env: EnvTypes, env_states: TensorState, joint_names: str | tuple[str]) -> torch.Tensor:
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


def base_height(env: EnvTypes, env_states: TensorState, target_height: float) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_height = robot_state.root_state[:, 2]
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
    """Reward alternating stance phases across feet following a target gait pattern."""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    command_name = "base_velocity"

    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = contact_forces.contact_forces_history[:, :, indices, :].norm(dim=-1).max(dim=1)[0] > 1.0
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
    contacts = contact_forces.contact_forces_history[:, :, indices, :].norm(dim=-1).max(dim=1)[0] > 1.0

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
    """Reward the swinging feet for clearing a specified height off the ground."""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    base = env_states.robots[env.name]
    foot_z_target_error = torch.square(base.body_state[:, indices, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(base.body_state[:, indices, 7:9], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std**2)


def _resolve_gait_params(
    env: EnvTypes,
    period: float | None,
    offset: list[float] | None,
    threshold: float | None,
    gait_key: str,
) -> tuple[float | None, list[float] | None, float | None]:
    if period is not None and offset is not None and threshold is not None:
        return period, offset, threshold

    params = None
    reward_entry = getattr(env, "reward_scales", {}).get(gait_key)
    if isinstance(reward_entry, tuple) and len(reward_entry) >= 2:
        params = reward_entry[1]

    if params is None:
        cfg_entry = getattr(getattr(env.cfg.rewards, "scales", None), gait_key, None)
        if isinstance(cfg_entry, tuple) and len(cfg_entry) >= 2:
            params = cfg_entry[1]

    if isinstance(params, dict):
        if period is None:
            period = params.get("period", period)
        if offset is None:
            offset = params.get("offset", offset)
        if threshold is None:
            threshold = params.get("threshold", threshold)

    return period, offset, threshold


def leg_raise_imitation(
    env: EnvTypes,
    env_states: TensorState,
    hip_joint_names: str | tuple[str],
    hip_min: float,
    std: float,
    max_lin_vel_cmd: float = 0.5,
    period: float | None = None,
    offset: list[float] | None = None,
    threshold: float | None = None,
    gait_key: str = "feet_gait",
) -> torch.Tensor:
    """Reward matching a knee/hip relationship on the front leg at low command ranges."""
    max_cmd = env.commands_manager.ranges.lin_vel_x[1]
    if max_cmd > max_lin_vel_cmd:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    period, offset, threshold = _resolve_gait_params(env, period, offset, threshold, gait_key)
    if period is None or offset is None or threshold is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    if period <= 0.0 or threshold >= 1.0:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    joint_names = env.sorted_joint_names
    hip_idx = _get_indices(env, hip_joint_names, joint_names)
    if hip_idx.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    num_legs = hip_idx.numel()

    joint_pos = env_states.robots[env.name].joint_pos
    hip_pos = joint_pos[:, hip_idx[:num_legs]]

    phase = ((env._episode_steps * env.step_dt) % period) / period
    offsets = torch.tensor(offset[:num_legs], device=env.device).view(1, -1)
    leg_phase = (phase.unsqueeze(1) + offsets) % 1.0
    swing_mask = leg_phase >= threshold
    swing_phase = (leg_phase - threshold) / max(1.0 - threshold, 1e-6)
    swing_phase = torch.clamp(swing_phase, min=0.0, max=1.0)
    swing_curve = torch.sin(torch.pi * swing_phase)

    hip_target = hip_min * swing_curve

    err = torch.square(hip_pos - hip_target)
    err = torch.where(swing_mask, err, torch.zeros_like(err))

    count = swing_mask.sum(dim=1)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    valid = count > 0
    if valid.any():
        mean_err = err.sum(dim=1)[valid] / count[valid]
        reward[valid] = torch.exp(-mean_err / std**2)
    return reward


def undesired_contacts(
    env: EnvTypes,
    env_states: TensorState,
    threshold: float,
    body_names: str | tuple[str] = "(?!.*ankle.*).*",
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)
    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = contact_forces.contact_forces_history[:, :, indices, :].norm(dim=-1).max(dim=1)[0] > threshold

    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def feet_air_time(
    env: EnvTypes,
    env_states: TensorState,
    threshold: float,
    body_names: str | tuple[str] = ".*_ankle_roll_link",
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds using contact duration."""
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names).long()
    num_feet = len(indices)
    if num_feet == 0:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    # Use a unique key for the state to avoid conflicts if multiple rewards use this
    state_key = f"feet_air_time_state_{hash_names(body_names)}"
    if state_key not in env.extras_buffer:
        env.extras_buffer[state_key] = {
            "air_time": torch.zeros(env.num_envs, num_feet, dtype=torch.float, device=env.device),
            "contact_time": torch.zeros(env.num_envs, num_feet, dtype=torch.float, device=env.device),
        }

    state = env.extras_buffer[state_key]
    air_time = state["air_time"]
    contact_time = state["contact_time"]

    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    # Check current contact status
    contact_now = contact_forces.contact_forces_history[:, -1, indices, :].norm(dim=-1) > 1.0

    # Update timers: reset air_time if contact, else increment. Reset contact_time if air, else increment.
    air_time[:] = torch.where(contact_now, torch.zeros_like(air_time), air_time + env.step_dt)
    contact_time[:] = torch.where(contact_now, contact_time + env.step_dt, torch.zeros_like(contact_time))

    # Single stance logic: Reward is based on valid single stance duration?
    # The reference implementation rewards maintaining single stance and accumulates time in mode.
    single_stance = torch.sum(contact_now.int(), dim=1) == 1
    # Time in current mode (either air or contact)
    in_mode_time = torch.where(contact_now, contact_time, air_time)

    # Reward is the minimum time in mode across feet, IF in single stance.
    # This implies we reward properly holding the stance/swing phase.
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    if hasattr(env, "reset_buf") and env.reset_buf.any():
        air_time[env.reset_buf] = 0.0
        contact_time[env.reset_buf] = 0.0

    return reward


def foot_parallel_to_ground(
    env: EnvTypes,
    env_states: TensorState,
    joint_names_lists: list[tuple[str, str, str]],
    std: float,
) -> torch.Tensor:
    """Reward for keeping the feet parallel only using joint positions of the legs.

    This function assumes that the sum of the hip pitch, knee pitch, and ankle pitch angles should be zero
    to keep the foot parallel to the base (and thus the ground, assuming a flat base).

    Args:
        env: The environment object.
        env_states: The state of the environment.
        joint_names_lists: A list of tuples, where each tuple contains the names of the hip pitch, knee pitch,
                           and ankle pitch joints for a leg, in that order.
                           Example: [("left_hip_pitch", "left_knee_pitch", "left_ankle_pitch"),
                                     ("right_hip_pitch", "right_knee_pitch", "right_ankle_pitch")]
        std: Standard deviation for the Gaussian kernel.

    Returns:
        The computed reward.
    """
    robot_state = env_states.robots[env.name]
    joint_pos = robot_state.joint_pos

    total_error = 0.0

    for hip_name, knee_name, ankle_name in joint_names_lists:
        hip_idx = _get_indices(env, hip_name, env.sorted_joint_names)
        knee_idx = _get_indices(env, knee_name, env.sorted_joint_names)
        ankle_idx = _get_indices(env, ankle_name, env.sorted_joint_names)

        hip_val = joint_pos[:, hip_idx].squeeze(-1)
        knee_val = joint_pos[:, knee_idx].squeeze(-1)
        ankle_val = joint_pos[:, ankle_idx].squeeze(-1)

        # Target ankle pitch is -(hip_pitch + knee_pitch)
        target_ankle_val = -(hip_val + knee_val)
        error = torch.square(ankle_val - target_ankle_val)
        total_error += error

    return torch.exp(-total_error / std**2)
