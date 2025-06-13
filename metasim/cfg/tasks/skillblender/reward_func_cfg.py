"""Reward functions for legged robot"""

from __future__ import annotations

import torch

from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.types import EnvState


# =====================reward functions=====================
def reward_lin_vel_z(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """Reward for z linear velocity."""
    return torch.square(states.robots[robot_name].extra["base_lin_vel"][:, 2])


def reward_ang_vel_xy(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """Reward for xy angular velocity."""
    return torch.sum(torch.square(states.robots[robot_name].extra["base_ang_vel"][:, :2]), dim=1)


def reward_orientation(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    quat_mismatch = torch.exp(
        -torch.sum(torch.abs(states.robots[robot_name].extra["base_euler_xyz"][:, :2]), dim=1) * 10
    )
    orientation = torch.exp(-torch.norm(states.robots[robot_name].extra["projected_gravity"][:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.0


def reward_base_height(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    # TODO check this reward formulation. It dose not match what is described in legged_gym
    """
    Penalize base height deviation from target.
    """
    stance_mask = states.robots[robot_name].extra["gait_phase"]
    measured_heights = torch.sum(
        states.robots[robot_name].body_state[:, cfg.feet_indices, 2] * stance_mask,
        dim=1,
    ) / torch.sum(stance_mask, dim=1)
    base_height = states.robots[robot_name].root_state[:, 2] - (measured_heights - 0.05)
    return torch.exp(-torch.abs(base_height - cfg.reward_cfg.base_height_target) * 100)


def reward_torques(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum(torch.square(states.robots[robot_name].joint_effort_target), dim=1)


def reward_dof_vel(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    return torch.sum(torch.square(states.robots[robot_name].joint_vel), dim=1)


def reward_dof_acc(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize high DOF accelerations.
    """
    return torch.sum(
        torch.square((states.robots[robot_name].extra["last_dof_vel"] - states.robots[robot_name].joint_vel) / cfg.dt),
        dim=1,
    )


def reward_action_rate(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize high action rate.
    """
    return torch.sum(
        torch.square(states.robots[robot_name].extra["last_actions"] - states.robots[robot_name].extra["actions"]),
        dim=1,
    )


def reward_collision(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_termination(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Reward for termination, used to reset the environment.
    """
    return states.robots[robot_name].extra["reset_buf"] * ~states.robots[robot_name].extra["time_out_buf"]


def reward_dof_pos_limits(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize DOF positions that are out of limits.
    """
    out_of_limits = -(states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def reward_dof_vel_limits(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_torque_limits(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_tracking_lin_vel(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Track linear velocity commands (xy axes).
    """
    lin_vel_diff = (
        states.robots[robot_name].extra["command"][:, :2] - states.robots[robot_name].extra["base_lin_vel"][:, :2]
    )
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error * cfg.reward_cfg.tracking_sigma), torch.mean(torch.abs(lin_vel_diff), dim=1)


def reward_tracking_ang_vel(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Track angular velocity commands (yaw).
    """
    ang_vel_diff = (
        states.robots[robot_name].extra["command"][:, 2] - states.robots[robot_name].extra["base_ang_vel"][:, 2]
    )
    ang_vel_error = torch.square(ang_vel_diff)
    return torch.exp(-ang_vel_error * cfg.reward_cfg.tracking_sigma), torch.abs(ang_vel_diff)


def reward_feet_air_time(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Calculates the reward for feet air time.
    """
    air_time = states.robots[robot_name].extra["feet_air_time"]
    return air_time.sum(dim=1)


def reward_stumble(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize stumbling based on contact forces.
    """
    return torch.any(
        torch.norm(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :2], dim=2)
        > 5 * torch.abs(states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2]),
        dim=1,
    )


# FIXME place default dof pos better
def reward_stand_still(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Reward for standing still, penalizing deviation from default joint positions.
    """
    return torch.sum(torch.abs(states.robots[robot_name].joint_pos - cfg.default_dof_pos), dim=1) * (
        torch.norm(states.robots[robot_name].extra["command"][:, :2], dim=1) < 0.1
    )


def reward_feet_contact_forces(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


# ==========================h1 walking========================
def reward_joint_pos(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Calculates the reward based on the difference between the current joint positions and the target joint positions.
    """
    joint_pos = states.robots[robot_name].joint_pos.clone()
    pos_target = states.robots[robot_name].extra["ref_dof_pos"].clone()
    diff = joint_pos - pos_target
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r, torch.mean(torch.abs(diff), dim=1)


def reward_feet_distance(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
    """
    foot_pos = states.robots[robot_name].body_state[:, cfg.feet_indices, :2]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    fd = cfg.reward_cfg.min_dist
    max_df = cfg.reward_cfg.max_dist
    d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, foot_dist


def reward_knee_distance(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


# @jhcao
def reward_elbow_distance(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_foot_slip(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Calculates the reward for minimizing foot slip.
    """
    contact = states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2] > 5.0
    foot_speed_norm = torch.norm(states.robots[robot_name].body_state[:, cfg.feet_indices, 10:12], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)


def reward_feet_contact_number(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Reward based on feet contact matching gait phase.
    """
    # TODO fix this
    contact = states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, 2] > 5.0
    stance_mask = states.robots[robot_name].extra["gait_phase"]
    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)


def reward_default_joint_pos(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Keep joint positions close to defaults (penalize yaw/roll).
    """
    joint_diff = states.robots[robot_name].joint_pos - cfg.default_joint_pd_target
    left_yaw_roll = joint_diff[:, :2]
    right_yaw_roll = joint_diff[:, 5:7]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)


def reward_upper_body_pos(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Keep upper body joints close to default positions.
    """
    torso_index = 10
    joint_diff = states.robots[robot_name].joint_pos - cfg.default_joint_pd_target
    upper_body_diff = joint_diff[:, torso_index:]  # start from torso
    upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
    return torch.exp(-4 * upper_body_error), upper_body_error


def reward_base_acc(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize base acceleration.
    """
    root_acc = states.robots[robot_name].extra["last_root_vel"] - states.robots[robot_name].root_state[:, 7:13]
    rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
    return rew


def reward_vel_mismatch_exp(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Penalize velocity mismatch.
    """
    lin_mismatch = torch.exp(-torch.square(states.robots[robot_name].extra["base_lin_vel"][:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(states.robots[robot_name].extra["base_ang_vel"][:, :2], dim=1) * 5.0)
    return (lin_mismatch + ang_mismatch) / 2.0


def reward_track_vel_hard(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_feet_clearance(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
    """
    Reward swing leg clearance.
    """
    return states.robots[robot_name].extra["feet_clearance"]


def reward_low_speed(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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


def reward_action_smoothness(states: EnvState, robot_name: str, cfg: BaseRLTaskCfg) -> torch.Tensor:
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
