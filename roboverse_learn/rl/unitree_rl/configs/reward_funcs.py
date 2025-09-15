"""Reward functions for legged robot"""

from __future__ import annotations

import torch

from metasim.types import DictEnvState


# =====================reward functions=====================
def reward_lin_vel_z(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """Reward for z linear velocity."""
    return torch.square(states.robots[robot_name].extra["base_lin_vel"][:, 2])


def reward_ang_vel_xy(states: DictEnvState, robot_name: str, cfg: BaseTaskCfg) -> torch.Tensor:
    xy = torch.norm(states.robots[robot_name].extra["base_ang_vel"][:, :2], dim=1)
    db = getattr(cfg.reward_cfg, "angvel_xy_deadband", 0.25)
    k = getattr(cfg.reward_cfg, "angvel_xy_k", 1.0)
    return torch.square(torch.clamp(xy - db, min=0.0)) * k


def reward_orientation(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    quat_mismatch = torch.exp(
        -torch.sum(torch.abs(states.robots[robot_name].extra["base_euler_xyz"][:, :2]), dim=1) * 10
    )
    orientation = torch.exp(-torch.norm(states.robots[robot_name].extra["projected_gravity"][:, :2], dim=1) * 20)
    allow_roll_gate = _vy_gate(states.robots[robot_name], getattr(cfg.reward_cfg, "upright_gate_vy", 0.12))
    return (quat_mismatch + orientation) / 2.0 * allow_roll_gate


def reward_orientation_sq(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    return torch.sum(torch.square(states.robots[robot_name].extra["projected_gravity"][:, :2]), dim=1)


def reward_base_height(states: DictEnvState, robot_name: str, cfg):
    base = states.robots[robot_name]
    z_root = base.root_state[:, 2]

    # Soft stance weights from contact forces (0..1), avoids hard switches
    F = torch.norm(base.extra["contact_forces"][:, cfg.feet_indices, :], dim=-1)  # (B, nfeet)
    w = torch.clamp((F - 20.0) / 40.0, 0.0, 1.0)  # tune 20/40 by your scale

    z_feet = states.robots[robot_name].body_state[:, cfg.feet_indices, 2]
    w_sum = w.sum(dim=1)
    z_foot_ref = (z_feet * w).sum(dim=1) / (w_sum + 1e-6)

    # fallback to min foot height if no clear stance
    z_foot_ref = torch.where(
        (w_sum > 0.2),
        z_foot_ref,
        z_feet.min(dim=1).values,
    )

    # clearance offset (was 0.05); keep small and configurable
    clearance = getattr(cfg.reward_cfg, "height_clearance", 0.03)
    h = z_root - (z_foot_ref - clearance)

    # target & shaping
    h_tgt = getattr(cfg.reward_cfg, "base_height_target", 0.9)
    err = h - h_tgt

    # Huber-ish: deadband then squared
    db = getattr(cfg.reward_cfg, "height_deadband", 0.015)  # 1.5 cm
    err_db = torch.clamp(torch.abs(err) - db, min=0.0)
    sigma = getattr(cfg.reward_cfg, "base_height_sigma", 0.04)  # 4 cm scale
    return torch.exp(-(err_db**2) / (2 * sigma**2))


def reward_base_height_sq(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    # Penalize base height away from target
    base_height = states.robots[robot_name].root_state[:, 2]
    return torch.square(base_height - cfg.reward_cfg.base_height_target)


def reward_torques(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum(torch.square(states.robots[robot_name].joint_effort_target), dim=1)


def reward_dof_vel(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    return torch.sum(torch.square(states.robots[robot_name].joint_vel), dim=1)


def reward_dof_acc(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high DOF accelerations.
    """
    return torch.sum(
        torch.square((states.robots[robot_name].extra["last_dof_vel"] - states.robots[robot_name].joint_vel) / (cfg.dt * cfg.decimation)),
        dim=1,
    )


def reward_action_rate(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high action rate.
    """
    return torch.sum(
        torch.square(states.robots[robot_name].extra["last_actions"] - states.robots[robot_name].extra["actions"]),
        dim=1,
    )


def reward_collision(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize collisions.
    """
    return torch.sum(
        1.0
        * (
            torch.norm(
                states.robots[robot_name].extra["contact_forces"][:, cfg.penalised_contact_indices, :],
                dim=-1,
            )
            > 0.1
        ),
        dim=1,
    )


def reward_termination(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward for termination, used to reset the environment.
    """
    return states.robots[robot_name].extra["reset_buf"] * ~states.robots[robot_name].extra["time_out_buf"]


def reward_dof_pos_limits(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize DOF positions that are out of limits.
    """
    out_of_limits = -(states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def reward_dof_vel_limits(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    return torch.sum(
        (
            torch.abs(states.robots[robot_name].dof_vel)
            - states.robots[robot_name].extra["dof_vel_limits"] * cfg.reward_cfg.soft_dof_vel_limit
        ).clip(min=0.0, max=1.0),
        dim=1,
    )


def reward_torque_limits(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum(
        (
            torch.abs(states.robots[robot_name].joint_effort_target)
            - cfg.torque_limits * cfg.reward_cfg.soft_torque_limit
        ).clip(min=0.0),
        dim=1,
    )


def reward_tracking_lin_vel(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Track linear velocity commands (xy axes).
    """
    lin_vel_diff = (
        states.robots[robot_name].extra["command"][:, :2] - states.robots[robot_name].extra["base_lin_vel"][:, :2]
    )
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error * cfg.reward_cfg.tracking_sigma), torch.mean(torch.abs(lin_vel_diff), dim=1)


def _vy_gate(base, thresh=0.12):
    if base.extra["command"].shape[1] <= 1:
        return torch.ones_like(base.extra["command"][:, 0])
    return (torch.abs(base.extra["command"][:, 1]) < thresh).float()


def reward_tracking_ang_vel(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Track angular velocity commands (yaw).
    """
    ang_vel_diff = (
        states.robots[robot_name].extra["command"][:, 2] - states.robots[robot_name].extra["base_ang_vel"][:, 2]
    )
    ang_vel_error = torch.square(ang_vel_diff)
    return torch.exp(-ang_vel_error * cfg.reward_cfg.tracking_sigma), torch.abs(ang_vel_diff)


# FIXME
def reward_feet_air_time(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Calculates the reward for feet air time.
    """
    air_time = states.robots[robot_name].extra["feet_air_time"]
    return air_time.sum(dim=1) * (torch.norm(states.robots[robot_name].extra["command"][:, :2], dim=1) > 0.1)


def reward_stumble(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize stumbling based on contact forces.
    """
    return torch.any(
        torch.norm(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :2], dim=2)
        > 5 * torch.abs(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2]),
        dim=1,
    )


# FIXME place default dof pos better
def reward_stand_still(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward for standing still, penalizing deviation from default joint positions.
    """
    return torch.sum(torch.abs(states.robots[robot_name].joint_pos - cfg.default_dof_pos), dim=1) * (
        torch.norm(states.robots[robot_name].extra["command"][:, :2], dim=1) < 0.1
    )


def reward_feet_contact_forces(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize high contact forces on feet.
    """
    return torch.sum(
        (
            torch.norm(
                states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :],
                dim=-1,
            )
            - cfg.reward_cfg.max_contact_force
        ).clip(0, 400),
        dim=1,
    )


def reward_contact(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    is_stance = states.robots[robot_name].extra["gait_phase"]
    # is_stance = states.robots[robot_name].extra["leg_phase"] < 0.55
    contact_forces = states.robots[robot_name].extra["contact_forces"]
    contact = contact_forces[:, cfg.feet_indices, 2] > 1
    # Reward when stance matches contact (True if both are the same)
    res = (contact == is_stance)
    return torch.sum(res, dim=1)


def reward_feet_swing_height(states: EnvState, robot_name: str, cfg: BaseTaskCfg) -> torch.Tensor:
    contact = torch.norm(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :3], dim=2) > 1.0
    pos_error = (
        torch.square(states.robots[robot_name].body_state[:, cfg.feet_indices, 2] - cfg.reward_cfg.target_feet_height)
        * ~contact
    )
    return torch.sum(pos_error, dim=(1))


def reward_alive(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    # Reward for staying alive
    return 1.0


def reward_contact_no_vel(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    # Penalize contact with no velocity
    contact = torch.norm(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :3], dim=2) > 1.0
    feet_state = states.robots[robot_name].body_state[:, cfg.feet_indices, :]
    feet_vel = feet_state[:, :, 7:10]
    contact_feet_vel = feet_vel * contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
    return torch.sum(penalize, dim=(1, 2))


def reward_hip_pos(states, robot_name, cfg):
    base = states.robots[robot_name]
    gate = _vy_gate(base, getattr(cfg.reward_cfg, "hip_pos_gate_vy", 0.12))
    dof_pos = base.joint_pos
    indices = torch.concat([cfg.left_yaw_roll_joint_indices, cfg.right_yaw_roll_joint_indices])
    dof_pos_hip = dof_pos[:, indices]
    return torch.sum(torch.square(dof_pos_hip), dim=1) * gate


# ==========================h1 walking========================
def reward_joint_pos(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Calculates the reward based on the difference between the current joint positions and the target joint positions.
    """
    joint_pos = states.robots[robot_name].joint_pos.clone()
    pos_target = states.robots[robot_name].extra["ref_dof_pos"].clone()
    diff = joint_pos - pos_target
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r, torch.mean(torch.abs(diff), dim=1)


def reward_feet_distance(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    base = states.robots[robot_name]
    feet_y = base.body_state[:, cfg.feet_indices, 1]  # (B, 2)
    step_width = torch.abs(feet_y[:, 0] - feet_y[:, 1])  # (B,)

    # Double support gating
    contact = base.extra["contact_forces"][:, cfg.feet_indices, 2] > 5.0
    both_stance = torch.all(contact, dim=1)

    # Step width band
    sw_min = getattr(cfg.reward_cfg, "min_dist", 0.18)
    sw_max = getattr(cfg.reward_cfg, "max_dist", 0.38)
    k = 100.0
    d_min = torch.clamp(step_width - sw_min, -0.5, 0.0)
    d_max = torch.clamp(step_width - sw_max, 0.0, 0.5)
    band = (torch.exp(-torch.abs(d_min) * k) + torch.exp(-torch.abs(d_max) * k)) / 2.0

    # Gate 1: Relax when there's lateral command (weaken step width constraint when vy_cmd is large)
    vy_cmd = base.extra["command"][:, 1] if base.extra["command"].shape[1] > 1 else 0.0
    vy_gate = getattr(cfg.reward_cfg, "sw_gate_vy", 0.2)  # m/s
    gate_cmd = 1.0 - torch.clamp(torch.abs(vy_cmd) / vy_gate, 0.0, 1.0)  # small vy→1, large vy→0

    # Gate 2: Relax when DCM error is large (don't constrain step width when "recovery" is needed)
    y = states.robots[robot_name].root_state[:, 1]
    vy = base.extra["base_lin_vel"][:, 1]
    z0 = getattr(cfg.reward_cfg, "base_height_target", 0.9)
    z0_t = torch.clamp(torch.as_tensor(z0, device=y.device, dtype=y.dtype), min=0.2)
    omega = torch.sqrt(torch.tensor(9.81, device=y.device, dtype=y.dtype) / z0_t)
    xi = y + vy / omega
    xi_ref = vy_cmd / omega
    dxi = torch.abs(xi - xi_ref)
    gate_dcm = torch.exp(-dxi * getattr(cfg.reward_cfg, "sw_dcm_relax", 8.0))  # large error→small gate

    gate = gate_cmd * gate_dcm  # Combined effect of both gates

    # Combination: only effective during double support; when gate is 0, degrades to 1.0 (neutral, doesn't affect other terms)
    raw = torch.where(both_stance, band, torch.ones_like(step_width))
    reward = gate * raw + (1 - gate) * torch.ones_like(step_width)

    return reward, step_width


def reward_knee_distance(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the knee of the humanoid.
    """
    knee_pos = states.robots[robot_name].body_state[:, cfg.knee_indices, :2]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    fd = cfg.reward_cfg.min_dist
    max_df = cfg.reward_cfg.max_dist / 2
    d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, knee_dist


def reward_elbow_distance(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the elbow of the humanoid.
    """
    elbow_pos = states.robots[robot_name].body_state[:, cfg.elbow_indices, :2]
    elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
    fd = cfg.reward_cfg.min_dist
    max_df = cfg.reward_cfg.max_dist / 2
    d_min = torch.clamp(elbow_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, elbow_dist


def reward_foot_slip(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Calculates the reward for minimizing foot slip.
    """
    contact = states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2] > 5.0
    foot_speed_norm = torch.norm(states.robots[robot_name].body_state[:, cfg.feet_indices, 10:12], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)


def reward_feet_contact_number(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward based on feet contact matching gait phase.
    """
    contact = states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2] > 5.0
    stance_mask = states.robots[robot_name].extra["gait_phase"]
    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)


def reward_default_joint_pos(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Keep joint positions close to defaults (penalize yaw/roll).
    """
    joint_diff = states.robots[robot_name].joint_pos - cfg.default_joint_pd_target
    left_yaw_roll = joint_diff[:, cfg.left_yaw_roll_joint_indices]
    right_yaw_roll = joint_diff[:, cfg.right_yaw_roll_joint_indices]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)


def reward_upper_body_pos(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Keep upper body joints close to default positions.
    """
    joint_diff = states.robots[robot_name].joint_pos - cfg.default_joint_pd_target
    upper_body_diff = joint_diff[:, cfg.upper_body_joint_indices]  # start from torso
    upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
    return torch.exp(-4 * upper_body_error), upper_body_error


def reward_base_acc(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize base acceleration.
    """
    root_acc = states.robots[robot_name].extra["last_root_vel"] - states.robots[robot_name].root_state[:, 7:13]
    rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
    return rew


def reward_vel_mismatch_exp(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize velocity mismatch.
    """
    lin_mismatch = torch.exp(-torch.square(states.robots[robot_name].extra["base_lin_vel"][:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(states.robots[robot_name].extra["base_ang_vel"][:, :2], dim=1) * 5.0)
    return (lin_mismatch + ang_mismatch) / 2.0


def reward_track_vel_hard(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Track linear and angular velocity commands.
    """
    lin_vel_error = torch.norm(
        states.robots[robot_name].extra["command"][:, :2] - states.robots[robot_name].extra["base_lin_vel"][:, :2],
        dim=1,
    )
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
    ang_vel_error = torch.abs(
        states.robots[robot_name].extra["command"][:, 2] - states.robots[robot_name].extra["base_ang_vel"][:, 2]
    )
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
    linear_error = 0.2 * (lin_vel_error + ang_vel_error)
    return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error


def reward_feet_clearance(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward swing leg clearance.
    """
    return states.robots[robot_name].extra["feet_clearance"]


def reward_low_speed(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize speed mismatch with command.
    """
    absolute_speed = torch.abs(states.robots[robot_name].extra["base_lin_vel"][:, 0])
    absolute_command = torch.abs(states.robots[robot_name].extra["command"][:, 0])
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)
    sign_mismatch = torch.sign(states.robots[robot_name].extra["base_lin_vel"][:, 0]) != torch.sign(
        states.robots[robot_name].extra["command"][:, 0]
    )
    reward = torch.zeros_like(states.robots[robot_name].extra["base_lin_vel"][:, 0])
    reward[speed_too_low] = -1.0
    reward[speed_too_high] = 0.0
    reward[speed_desired] = 1.2
    reward[sign_mismatch] = -2.0
    return reward * (states.robots[robot_name].extra["command"][:, 0].abs() > 0.1)


def reward_action_smoothness(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Penalize jerk in actions.
    """
    term_1 = torch.sum(
        torch.square(states.robots[robot_name].extra["last_actions"] - states.robots[robot_name].extra["actions"]),
        dim=1,
    )
    term_2 = torch.sum(
        torch.square(
            states.robots[robot_name].extra["actions"]
            + states.robots[robot_name].extra["last_last_actions"]
            - 2 * states.robots[robot_name].extra["last_actions"]
        ),
        dim=1,
    )
    term_3 = 0.05 * torch.sum(torch.abs(states.robots[robot_name].extra["actions"]), dim=1)
    return term_1 + term_2 + term_3


def reward_waist_joint_stability(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward for keeping waist joints (yaw, roll, pitch) stable and close to default positions.
    This directly penalizes waist joint deviations and velocities to prevent shaking.
    """
    joint_pos = states.robots[robot_name].joint_pos
    joint_vel = states.robots[robot_name].joint_vel

    waist_indices = cfg.waist_joint_indices

    # Get waist joint positions and velocities
    waist_pos = joint_pos[:, waist_indices]
    waist_vel = joint_vel[:, waist_indices]

    # Default waist positions (should be close to 0 for stability)
    waist_default = cfg.default_joint_pd_target[:, waist_indices]

    # Penalize deviation from default positions
    pos_error = torch.norm(waist_pos - waist_default, dim=1)
    pos_penalty = torch.exp(-pos_error * 20.0)

    # Penalize high waist joint velocities
    vel_error = torch.norm(waist_vel, dim=1)
    vel_penalty = torch.exp(-vel_error * 15.0)

    # Combine position and velocity penalties
    waist_stability_reward = 0.6 * pos_penalty + 0.4 * vel_penalty

    return waist_stability_reward


def reward_hip_upright_axis(states: DictEnvState, robot_name: str, cfg) -> torch.Tensor:
    """
    Reward for keeping hip/pelvis axis oriented upward (vertical).
    This penalizes hip tilting and rolling motions that cause shaking.

    Uses the pelvis/hip body orientation to ensure the local Z-axis stays aligned with world Z-axis.
    """
    # Get hip/pelvis body indices - typically the torso or pelvis link
    if hasattr(cfg, 'torso_indices') and len(cfg.torso_indices) > 0:
        hip_body_idx = cfg.torso_indices[0]  # Use first torso link as hip reference
    else:
        # Fallback to base body (root link) if no torso indices defined
        hip_body_idx = 0

    # Get body state for the hip/pelvis
    body_quat = states.robots[robot_name].body_state[:, hip_body_idx, 3:7]  # quaternion [w, x, y, z]

    # Convert quaternion to rotation matrix to get local Z-axis direction
    # Local Z-axis in world coordinates after rotation
    w, x, y, z = body_quat[:, 0], body_quat[:, 1], body_quat[:, 2], body_quat[:, 3]

    # Extract local Z-axis (3rd column of rotation matrix)
    # R[2,2] = 1 - 2*(x^2 + y^2) - this is the Z-component of local Z-axis in world frame
    local_z_world_z = 1 - 2 * (x**2 + y**2)

    # We want local Z-axis to be aligned with world Z-axis (pointing up)
    # Perfect alignment: local_z_world_z = 1, worst case: local_z_world_z = -1
    alignment_error = 1.0 - local_z_world_z  # Error ranges from 0 (perfect) to 2 (upside down)

    # Use exponential reward function - higher reward for better alignment
    hip_upright_reward = torch.exp(-alignment_error * 5.0)  # Scale factor 5.0 for sensitivity

    return hip_upright_reward
