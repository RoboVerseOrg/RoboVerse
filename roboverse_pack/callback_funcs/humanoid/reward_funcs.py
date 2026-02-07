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


def _range_hits_limit(
    curr_range: tuple[float, float] | list[float], limit_range: tuple[float, float] | list[float], atol: float = 1.0e-6
) -> bool:
    curr_lo, curr_hi = float(curr_range[0]), float(curr_range[1])
    limit_lo, limit_hi = float(limit_range[0]), float(limit_range[1])
    return curr_lo <= (limit_lo + atol) and curr_hi >= (limit_hi - atol)


def _is_max_velocity_curriculum(env: EnvTypes, atol: float = 1.0e-6) -> bool:
    curriculum_cfg = getattr(getattr(env, "cfg", None), "curriculum", None)
    if curriculum_cfg is None or not getattr(curriculum_cfg, "enabled", False):
        return False

    commands_manager = getattr(env, "commands_manager", None)
    if commands_manager is None:
        return False

    ranges = getattr(commands_manager, "ranges", None)
    limit_ranges = getattr(commands_manager, "limit_ranges", None)
    if ranges is None or limit_ranges is None:
        return False

    required_axes = ("lin_vel_x", "lin_vel_y")
    for axis in required_axes:
        if not hasattr(ranges, axis) or not hasattr(limit_ranges, axis):
            return False
        if not _range_hits_limit(getattr(ranges, axis), getattr(limit_ranges, axis), atol=atol):
            return False

    return True


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
    phase_offset: float = 0.275,
    body_names: str | tuple[str] = ".*ankle_roll.*",
) -> torch.Tensor:
    """Reward alternating stance phases across feet following a target gait pattern.

    Args:
        env: The environment object.
        env_states: The state of the environment.
        period: Gait period in seconds.
        offset: Phase offsets for each leg (e.g., [0.0, 0.5] for alternating).
        threshold: Stance/swing threshold (stance when phase < threshold).
        phase_offset: Shift to align with hip trajectory (0.25 aligns with sin-based trajectory).
        body_names: Regex pattern for foot bodies.
    """
    indices = _get_indices(env, body_names, env_states.robots[env.name].body_names)

    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = contact_forces.contact_forces_history[:, :, indices, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # Compute leg phases with offset
    global_phase = ((env._episode_steps * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        # Apply phase_offset to shift expected contact timing
        phase = (global_phase + offset_ + phase_offset) % 1.0
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


def _compute_sine_flat_trajectory(phase: torch.Tensor, static_duration: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes a flat-top sine trajectory and its velocity profile (shape).

    The trajectory consists of a Rise -> Plateau -> Fall -> (gap) pattern.

    Returns:
        trajectory: The position trajectory (sign * magnitude).
        velocity_profile: The magnitude of the derivative (always positive), zero during plateau.
    """
    # Lead Start: 0.5 + static_duration
    lead_start = 0.5 + static_duration
    p_rel = (phase - lead_start) % 1.0

    # Lead Duration is 0.5 (Half Cycle).
    is_lead = p_rel < 0.5

    # Within Lead (or Lag), we have Rise -> Plateau -> Fall. Total Duration 0.5.
    plateau_len = static_duration
    rise_len = (0.5 - plateau_len) / 2.0

    # Local phase within the 0.5 window
    # For Lag, we use the second half (0.5 to 1.0) mapped to 0 to 0.5
    local_p = torch.where(is_lead, p_rel, p_rel - 0.5)

    mag = torch.zeros_like(phase)
    vel = torch.zeros_like(phase)

    in_rise = local_p < rise_len
    in_plateau = (local_p >= rise_len) & (local_p < (rise_len + plateau_len))
    in_fall = local_p >= (rise_len + plateau_len)

    # Rise: Sin(0 to pi/2)
    rise_arg = local_p / rise_len
    mag = torch.where(in_rise, torch.sin(0.5 * torch.pi * rise_arg), mag)
    vel = torch.where(in_rise, torch.cos(0.5 * torch.pi * rise_arg), vel)

    # Plateau: 1.0 (pos), 0.0 (vel)
    mag = torch.where(in_plateau, torch.ones_like(mag), mag)
    # vel remains 0

    # Fall: Sin(pi/2 to pi)
    # Distance from end (0.5)
    dist_from_end = 0.5 - local_p
    fall_arg = dist_from_end / rise_len
    mag = torch.where(in_fall, torch.sin(0.5 * torch.pi * fall_arg), mag)
    vel = torch.where(in_fall, torch.cos(0.5 * torch.pi * fall_arg), vel)

    # Sign: Lead is negative trajectory (inverted per user request), Lag is positive
    sign = torch.where(is_lead, -torch.ones_like(phase), torch.ones_like(phase))

    return sign * mag, vel


def leg_raise_imitation(
    env: EnvTypes,
    env_states: TensorState,
    hip_pitch_names: str | tuple[str],
    hip_roll_names: str | tuple[str],
    knee_names: str | tuple[str],
    hip_pitch_amplitude: float,
    hip_roll_amplitude: float,
    knee_coupling: float,
    std: float,
    period: float,
    offset: list[float],
    static_duration: float = 0.05,
    velocity_scale: float = 1.0,
    phase_offset: float = 0.0,
    disable_at_max_curriculum: bool = False,
) -> torch.Tensor:
    """Command-adaptive leg raise imitation with full cycle supervision.

    This reward guides hip pitch, hip roll, and knee joints throughout the entire gait cycle:
    - Hip pitch follows a sinusoidal trajectory, amplitude scales with forward velocity (cmd_x)
    - Hip roll target scales with lateral velocity (cmd_y) for side-stepping
    - Knee bends during lead phase (derived from hip pitch), stays straight during lag
    - Knee target is derived from the velocity profile of the hip pitch trajectory (pi/2 shift)

    Args:
        env: The environment object.
        env_states: The state of the environment.
        hip_pitch_names: Names of hip pitch joints (e.g., left_hip_pitch, right_hip_pitch).
        hip_roll_names: Names of hip roll joints (e.g., left_hip_roll, right_hip_roll).
        knee_names: Names of knee joints.
        hip_pitch_amplitude: Base amplitude for hip pitch swing (negative = forward).
        hip_roll_amplitude: Base amplitude for hip roll during lateral movement.
        knee_coupling: Factor to derive knee bend from hip pitch (e.g., 0.5-0.7).
        std: Standard deviation for the Gaussian kernel.
        period: Gait period in seconds.
        offset: Phase offsets for each leg (e.g., [0.0, 0.5] for alternating).
        static_duration: Duration of the static hold for both feet.
        velocity_scale: Multiplier for how much command velocity affects hip pitch amplitude.
        phase_offset: Shift to align with gait pattern.
        disable_at_max_curriculum: If True, return zero reward when the velocity curriculum is at maximum.

    Returns:
        The computed reward.
    """
    if period <= 0.0 or static_duration >= 0.25:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    if disable_at_max_curriculum and _is_max_velocity_curriculum(env):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    joint_names = env.sorted_joint_names
    hip_pitch_idx = _get_indices(env, hip_pitch_names, joint_names)
    hip_roll_idx = _get_indices(env, hip_roll_names, joint_names)
    knee_idx = _get_indices(env, knee_names, joint_names)
    if hip_pitch_idx.numel() == 0 or hip_roll_idx.numel() == 0 or knee_idx.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    num_legs = min(hip_pitch_idx.numel(), hip_roll_idx.numel(), knee_idx.numel())

    joint_pos = env_states.robots[env.name].joint_pos
    hip_pitch_pos = joint_pos[:, hip_pitch_idx[:num_legs]]
    hip_roll_pos = joint_pos[:, hip_roll_idx[:num_legs]]
    knee_pos = joint_pos[:, knee_idx[:num_legs]]

    # Get command velocities
    cmd_x = env.commands_manager.value[:, 0:1]  # Forward velocity (N, 1)
    cmd_y = env.commands_manager.value[:, 1:2]  # Lateral velocity (N, 1)
    cmd_magnitude = torch.sqrt(cmd_x**2 + cmd_y**2)  # Overall command magnitude (N, 1)

    # Command-adaptive hip pitch amplitude:
    # - Direction: determined by sign of cmd_x (forward = negative hip pitch, backward = positive)
    # - Magnitude: scales with |cmd_x|
    # When cmd_x is near zero, default to forward direction (sign = 1)
    direction = torch.sign(cmd_x)
    direction = torch.where(torch.abs(cmd_x) < 0.05, torch.ones_like(direction), direction)
    pitch_magnitude = 1.0 + velocity_scale * torch.abs(cmd_x)  # (N, 1)
    # hip_pitch_amplitude is typically negative (forward swing), direction inverts for backward
    adapted_pitch_amplitude = hip_pitch_amplitude * direction * pitch_magnitude  # (N, 1)

    # Compute phase for each leg
    global_phase = ((env._episode_steps * env.step_dt) % period) / period
    offsets_tensor = torch.tensor(offset[:num_legs], device=env.device).view(1, -1)
    # Apply phase_offset to shift the entire cycle
    # User requested to move curve 0.05 lefter (total) => Add 0.05 to phase
    leg_phase = (global_phase.unsqueeze(1) + offsets_tensor + phase_offset + 0.05) % 1.0  # (N, num_legs)

    trajectory, velocity_profile = _compute_sine_flat_trajectory(leg_phase, static_duration)

    # Targets (always active, no static phases)
    hip_pitch_target = adapted_pitch_amplitude * trajectory

    # Hip roll: during both lead and lag, direction based on cmd_y and leg side
    leg_sign = torch.tensor([1.0, -1.0][:num_legs], device=env.device).view(1, -1)
    hip_roll_target = hip_roll_amplitude * cmd_y * leg_sign * trajectory

    # Knee: scales with overall command magnitude (both forward and lateral movement)
    # This ensures knee bending during side-stepping as well as forward/backward walking.
    # The velocity_profile provides the phase-dependent shape (cos pulse during rise/fall, 0 during plateau).
    knee_amplitude = 1.0 + velocity_scale * cmd_magnitude  # Scale with total velocity
    knee_target = knee_coupling * abs(hip_pitch_amplitude) * knee_amplitude * velocity_profile

    # Compute errors for all joints
    hip_pitch_err = torch.square(hip_pitch_pos - hip_pitch_target)
    hip_roll_err = torch.square(hip_roll_pos - hip_roll_target)
    knee_err = torch.square(knee_pos - knee_target)
    total_err = hip_pitch_err + hip_roll_err + knee_err

    # Mean error across legs
    mean_err = total_err.mean(dim=1)
    reward = torch.exp(-mean_err / std**2)

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
    double_flight_penalty: float = 0.0,
    period: float | None = None,
    offset: list[float] | None = None,
    gait_threshold: float = 0.55,
    phase_offset: float = 0.275,
    contact_force_threshold: float = 1.0,
    body_names: str | tuple[str] = ".*_ankle_roll_link",
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds using contact duration.

    Args:
        env: The environment object.
        env_states: The state of the environment.
        threshold: Maximum air time to reward.
        double_flight_penalty: Penalty when both feet are in air (positive value = penalty).
        period: Optional gait period (seconds). When provided alongside `offset`, the reward is
            phase-gated to only reward air time of the expected swing foot.
        offset: Optional phase offsets for each foot. When provided alongside `period`, the reward is
            phase-gated to only reward air time of the expected swing foot.
        gait_threshold: Stance/swing threshold in normalized phase, matching `feet_gait` (stance when phase < threshold).
        phase_offset: Phase offset applied before thresholding, matching `feet_gait` / `leg_raise_imitation` alignment.
        contact_force_threshold: Contact force norm threshold for contact detection.
        body_names: Regex pattern for foot bodies.
    """
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
    contact_now = contact_forces.contact_forces_history[:, -1, indices, :].norm(dim=-1) > contact_force_threshold

    if period is not None and offset is not None:
        if period <= 0.0:
            return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

        # Phase-gated variant (aligned with `feet_gait` and `leg_raise_imitation` timings):
        # Only count / reward air time for the foot that is expected to be in swing.
        global_phase = ((env._episode_steps * env.step_dt) % period) / period  # (N,)

        offset_list = list(offset)
        if len(offset_list) < num_feet:
            offset_list = offset_list + [0.0] * (num_feet - len(offset_list))

        offsets_tensor = torch.tensor(offset_list[:num_feet], device=env.device).view(1, -1)
        foot_phase = (global_phase.unsqueeze(1) + offsets_tensor + phase_offset) % 1.0  # (N, num_feet)
        expected_stance = foot_phase < gait_threshold
        expected_swing = ~expected_stance

        # Update timers (only within expected mode to prevent "banking" time outside the intended window).
        air_time[:] = torch.where(
            expected_swing & (~contact_now),
            air_time + env.step_dt,
            torch.zeros_like(air_time),
        )
        contact_time[:] = torch.where(
            expected_stance & contact_now,
            contact_time + env.step_dt,
            torch.zeros_like(contact_time),
        )

        reward = torch.clamp(air_time.sum(dim=1), max=threshold)
        double_flight = contact_now.sum(dim=1) == 0
        reward = torch.where(double_flight, -double_flight_penalty * torch.ones_like(reward), reward)

        if hasattr(env, "reset_buf") and env.reset_buf.any():
            air_time[env.reset_buf] = 0.0
            contact_time[env.reset_buf] = 0.0

        return reward

    # Default (phase-agnostic) variant.
    air_time[:] = torch.where(contact_now, torch.zeros_like(air_time), air_time + env.step_dt)
    contact_time[:] = torch.where(contact_now, contact_time + env.step_dt, torch.zeros_like(contact_time))

    # Count feet in contact
    num_contacts = contact_now.sum(dim=1)
    single_stance = num_contacts == 1
    double_flight = num_contacts == 0

    # Time in current mode (either air or contact)
    in_mode_time = torch.where(contact_now, contact_time, air_time)

    # Reward for single stance, penalty for double flight
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    # Apply penalty when both feet are in air
    reward = torch.where(double_flight, -double_flight_penalty * torch.ones_like(reward), reward)

    if hasattr(env, "reset_buf") and env.reset_buf.any():
        air_time[env.reset_buf] = 0.0
        contact_time[env.reset_buf] = 0.0

    return reward


def foot_parallel_to_ground(
    env: EnvTypes,
    env_states: TensorState,
    joint_names_lists: list[tuple[str, str, str]],
    std: float,
    disable_at_max_curriculum: bool = False,
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
        disable_at_max_curriculum: If True, return zero reward when the velocity curriculum is at maximum.

    Returns:
        The computed reward.
    """
    if disable_at_max_curriculum and _is_max_velocity_curriculum(env):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

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
