from __future__ import annotations

import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.anymal_cfg import AnymalCfg as AnymalRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass
from metasim.utils.math import (
    quat_rotate_inverse,
)

from ..base_task_cfg import BaseTaskCfg


@configclass
class AnymalTerrainCfg(BaseTaskCfg):
    episode_length = 1000
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    terrain_type = "plane"
    terrain_curriculum = True
    terrain_num_levels = 5
    terrain_num_terrains = 8
    terrain_map_length = 8.0
    terrain_map_width = 8.0
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    terrain_static_friction = 1.0
    terrain_dynamic_friction = 1.0
    terrain_restitution = 0.0
    terrain_slope_threshold = 0.75

    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    height_meas_scale = 5.0
    action_scale = 0.5

    terminal_reward = -0.0
    lin_vel_xy_reward_scale = 1.0
    lin_vel_z_reward_scale = -4.0
    ang_vel_z_reward_scale = 0.5
    ang_vel_xy_reward_scale = -0.05
    orient_reward_scale = -0.0
    torque_reward_scale = -0.00001
    joint_acc_reward_scale = -0.0005
    base_height_reward_scale = -0.0
    feet_air_time_reward_scale = 1.0
    knee_collision_reward_scale = -0.25
    feet_stumble_reward_scale = -0.0
    action_rate_reward_scale = -0.01
    hip_reward_scale = -0.0

    command_x_range = [-1.0, 1.0]
    command_y_range = [-1.0, 1.0]
    command_yaw_range = [-1.0, 1.0]

    base_init_state = [0.0, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    default_joint_angles = {
        "LF_HAA": 0.03,
        "LH_HAA": 0.03,
        "RF_HAA": -0.03,
        "RH_HAA": -0.03,
        "LF_HFE": 0.4,
        "LH_HFE": -0.4,
        "RF_HFE": 0.4,
        "RH_HFE": -0.4,
        "LF_KFE": -0.8,
        "LH_KFE": 0.8,
        "RF_KFE": -0.8,
        "RH_KFE": 0.8,
    }

    decimation = 4
    control_frequency_inv = 1
    push_interval_s = 15.0
    allow_knee_contacts = False

    kp = 50.0
    kd = 2.0

    add_noise = True
    noise_level = 1.0
    linear_velocity_noise = 1.5
    angular_velocity_noise = 0.2
    gravity_noise = 0.05
    dof_position_noise = 0.01
    dof_velocity_noise = 1.5
    height_measurement_noise = 0.1

    friction_range = [0.5, 1.25]

    robot: AnymalRobotCfg = AnymalRobotCfg()

    objects: list[RigidObjCfg] = []

    control: ControlCfg = ControlCfg(action_scale=0.5, action_offset=True, torque_limit_scale=1.0)

    checker = EmptyChecker()

    observation_space = {"shape": [188]}

    def __post_init__(self):
        super().__post_init__()

        dt = self.decimation * 0.005
        self.terminal_reward *= dt
        self.lin_vel_xy_reward_scale *= dt
        self.lin_vel_z_reward_scale *= dt
        self.ang_vel_z_reward_scale *= dt
        self.ang_vel_xy_reward_scale *= dt
        self.orient_reward_scale *= dt
        self.torque_reward_scale *= dt
        self.joint_acc_reward_scale *= dt
        self.base_height_reward_scale *= dt
        self.feet_air_time_reward_scale *= dt
        self.knee_collision_reward_scale *= dt
        self.feet_stumble_reward_scale *= dt
        self.action_rate_reward_scale *= dt
        self.hip_reward_scale *= dt

        self._commands = None
        self._actions = None
        self._last_actions = None
        self._last_dof_vel = None
        self._feet_air_time = None
        self._push_counter = 0
        self._terrain_levels = None
        self._terrain_types = None
        self._height_points = None
        self._measured_heights = None
        self._episode_sums = None
        self._noise_scale_vec = None

    def get_observation(self, states):
        observations = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["anymal"]

            base_quat = torch.tensor(robot_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
            base_lin_vel = torch.tensor(
                robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32, device=device
            )
            base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
            base_pos = torch.tensor(robot_state.get("pos", [0.0, 0.0, 0.62]), dtype=torch.float32, device=device)

            base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel) * self.lin_vel_scale
            base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel) * self.ang_vel_scale

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
            projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

            if "joint_qpos" in robot_state:
                dof_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32, device=device)
            elif "dof_pos" in robot_state:
                dof_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32, device=device)
            else:
                dof_pos = torch.zeros(12, dtype=torch.float32, device=device)

            dof_pos_scaled = dof_pos * self.dof_pos_scale

            if "joint_qvel" in robot_state:
                dof_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32, device=device)
            elif "dof_vel" in robot_state:
                dof_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32, device=device)
            else:
                dof_vel = torch.zeros(12, dtype=torch.float32, device=device)

            dof_vel_scaled = dof_vel * self.dof_vel_scale

            if self._commands is not None and i < len(self._commands):
                commands = self._commands[i][:3].to(device)
            else:
                commands = torch.zeros(3, dtype=torch.float32, device=device)

            commands_scaled = commands * torch.tensor(
                [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=device
            )

            if self._measured_heights is not None and i < len(self._measured_heights):
                heights = self._measured_heights[i].to(device)
            else:
                heights = torch.zeros(140, dtype=torch.float32, device=device)

            heights_scaled = torch.clip(base_pos[2] - 0.5 - heights, -1, 1.0) * self.height_meas_scale

            if self._actions is not None and i < len(self._actions):
                actions = self._actions[i].to(device)
            else:
                actions = torch.zeros(12, dtype=torch.float32, device=device)

            obs = torch.cat([
                base_lin_vel_base,
                base_ang_vel_base,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
                heights_scaled,
                actions,
            ])

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 188))

    def reward_fn(self, states, actions):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["anymal"]

            base_quat = robot_state.root_state[:, 3:7]
            base_pos = robot_state.root_state[:, :3]
            base_lin_vel = robot_state.root_state[:, 7:10]
            base_ang_vel = robot_state.root_state[:, 10:13]

            base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel)
            base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel)

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=base_pos.device).repeat(base_pos.shape[0], 1)
            projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

            if self._commands is not None:
                commands = self._commands
            else:
                num_envs = base_quat.shape[0]
                commands = torch.zeros((num_envs, 4), device=base_quat.device)

            lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_base[:, :2]), dim=1)
            ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_base[:, 2])
            rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.lin_vel_xy_reward_scale
            rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.ang_vel_z_reward_scale

            rew_lin_vel_z = torch.square(base_lin_vel_base[:, 2]) * self.lin_vel_z_reward_scale
            rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel_base[:, :2]), dim=1) * self.ang_vel_xy_reward_scale

            rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.orient_reward_scale

            rew_base_height = torch.square(base_pos[:, 2] - 0.52) * self.base_height_reward_scale

            if hasattr(robot_state, "torques"):
                torques = robot_state.torques
            else:
                torques = torch.zeros((base_quat.shape[0], 12), device=base_quat.device)
            rew_torque = torch.sum(torch.square(torques), dim=1) * self.torque_reward_scale

            dof_vel = robot_state.joint_vel
            if self._last_dof_vel is not None:
                rew_joint_acc = (
                    torch.sum(torch.square(self._last_dof_vel - dof_vel), dim=1) * self.joint_acc_reward_scale
                )
            else:
                rew_joint_acc = torch.zeros(base_quat.shape[0], device=base_quat.device)

            if hasattr(robot_state, "contact_forces"):
                contact_forces = robot_state.contact_forces

                knee_contact = torch.zeros(base_quat.shape[0], dtype=torch.bool, device=base_quat.device)
                rew_collision = knee_contact.float() * self.knee_collision_reward_scale

                feet_contact = contact_forces[:, -4:, 2] > 1.0
                if self._feet_air_time is not None:
                    first_contact = (self._feet_air_time > 0.0) * feet_contact
                    self._feet_air_time += 0.02
                    rew_air_time = (
                        torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1) * self.feet_air_time_reward_scale
                    )
                    rew_air_time *= torch.norm(commands[:, :2], dim=1) > 0.1
                    self._feet_air_time *= ~feet_contact
                else:
                    rew_air_time = torch.zeros(base_quat.shape[0], device=base_quat.device)

                stumble = (torch.norm(contact_forces[:, -4:, :2], dim=2) > 5.0) * (
                    torch.abs(contact_forces[:, -4:, 2]) < 1.0
                )
                rew_stumble = torch.sum(stumble, dim=1) * self.feet_stumble_reward_scale
            else:
                rew_collision = torch.zeros(base_quat.shape[0], device=base_quat.device)
                rew_air_time = torch.zeros(base_quat.shape[0], device=base_quat.device)
                rew_stumble = torch.zeros(base_quat.shape[0], device=base_quat.device)

            if self._last_actions is not None:
                rew_action_rate = (
                    torch.sum(torch.square(self._last_actions - self._actions), dim=1) * self.action_rate_reward_scale
                )
            else:
                rew_action_rate = torch.zeros(base_quat.shape[0], device=base_quat.device)

            dof_pos = robot_state.joint_pos
            hip_indices = [0, 3, 6, 9]
            default_hip_pos = torch.tensor([0.03, 0.03, -0.03, -0.03], device=dof_pos.device, dtype=dof_pos.dtype)
            rew_hip = torch.sum(torch.abs(dof_pos[:, hip_indices] - default_hip_pos), dim=1) * self.hip_reward_scale

            total_reward = (
                rew_lin_vel_xy
                + rew_ang_vel_z
                + rew_lin_vel_z
                + rew_ang_vel_xy
                + rew_orient
                + rew_base_height
                + rew_torque
                + rew_joint_acc
                + rew_collision
                + rew_action_rate
                + rew_air_time
                + rew_hip
                + rew_stumble
            )

            total_reward = torch.clip(total_reward, min=0.0, max=None)

            if hasattr(self, "_terminations"):
                total_reward += self.terminal_reward * self._terminations.float()

            if self._episode_sums is not None:
                self._episode_sums["lin_vel_xy"] += rew_lin_vel_xy
                self._episode_sums["ang_vel_z"] += rew_ang_vel_z
                self._episode_sums["lin_vel_z"] += rew_lin_vel_z
                self._episode_sums["ang_vel_xy"] += rew_ang_vel_xy
                self._episode_sums["orient"] += rew_orient
                self._episode_sums["torques"] += rew_torque
                self._episode_sums["joint_acc"] += rew_joint_acc
                self._episode_sums["collision"] += rew_collision
                self._episode_sums["stumble"] += rew_stumble
                self._episode_sums["action_rate"] += rew_action_rate
                self._episode_sums["air_time"] += rew_air_time
                self._episode_sums["base_height"] += rew_base_height
                self._episode_sums["hip"] += rew_hip

            return total_reward

        else:
            rewards = []
            for i, env_state in enumerate(states):
                rewards.append(1.0)
            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["anymal"]

            terminations = torch.zeros(
                robot_state.root_state.shape[0], dtype=torch.bool, device=robot_state.root_state.device
            )

            if hasattr(robot_state, "contact_forces"):
                contact_forces = robot_state.contact_forces
                base_contact = torch.norm(contact_forces[:, 0, :], dim=1) > 1.0
                terminations |= base_contact

                if not self.allow_knee_contacts:
                    pass

            self._terminations = terminations
            return terminations

        else:
            terminations = []
            for env_state in states:
                terminations.append(False)
            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        self._commands = None
        self._actions = None
        self._last_actions = None
        self._last_dof_vel = None
        self._feet_air_time = None
        self._push_counter = 0
        self._terrain_levels = None
        self._terrain_types = None
        self._height_points = None
        self._measured_heights = None
        self._episode_sums = None

    def reset(self, env_ids=None):
        if env_ids is None:
            num_envs = 1
        else:
            num_envs = len(env_ids)

        device = (
            self._commands.device
            if self._commands is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        commands = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        commands[:, 0] = (
            torch.rand(num_envs, device=device) * (self.command_x_range[1] - self.command_x_range[0])
            + self.command_x_range[0]
        )
        commands[:, 1] = (
            torch.rand(num_envs, device=device) * (self.command_y_range[1] - self.command_y_range[0])
            + self.command_y_range[0]
        )
        commands[:, 2] = (
            torch.rand(num_envs, device=device) * (self.command_yaw_range[1] - self.command_yaw_range[0])
            + self.command_yaw_range[0]
        )
        commands[:, 3] = 0.0

        commands *= (torch.norm(commands[:, :2], dim=1) > 0.25).unsqueeze(1)

        if self._commands is None:
            self._commands = commands
        else:
            if env_ids is not None and len(env_ids) > 0:
                for i, env_id in enumerate(env_ids):
                    if env_id < len(self._commands):
                        self._commands[env_id] = commands[i]
            else:
                self._commands = commands

        if env_ids is not None and len(env_ids) > 0:
            if self._last_actions is not None:
                for env_id in env_ids:
                    if env_id < len(self._last_actions):
                        self._last_actions[env_id] = 0.0
            if self._last_dof_vel is not None:
                for env_id in env_ids:
                    if env_id < len(self._last_dof_vel):
                        self._last_dof_vel[env_id] = 0.0
            if self._feet_air_time is not None:
                for env_id in env_ids:
                    if env_id < len(self._feet_air_time):
                        self._feet_air_time[env_id] = 0.0
            if self._episode_sums is not None:
                for key in self._episode_sums.keys():
                    for env_id in env_ids:
                        if env_id < len(self._episode_sums[key]):
                            self._episode_sums[key][env_id] = 0.0

    def post_reset(self):
        pass

    def initialize_buffers(self, num_envs, device):
        self._commands = torch.zeros((num_envs, 4), device=device)

        self._actions = torch.zeros((num_envs, 12), device=device)
        self._last_actions = torch.zeros((num_envs, 12), device=device)

        self._last_dof_vel = torch.zeros((num_envs, 12), device=device)

        self._feet_air_time = torch.zeros((num_envs, 4), device=device)

        if self.terrain_curriculum:
            max_init_level = self.terrain_num_levels - 1
        else:
            max_init_level = 0
        self._terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=device)
        self._terrain_types = torch.randint(0, self.terrain_num_terrains, (num_envs,), device=device)

        self._init_height_points(device)
        self._measured_heights = torch.zeros((num_envs, 140), device=device)

        self._episode_sums = {}
        for key in [
            "lin_vel_xy",
            "lin_vel_z",
            "ang_vel_z",
            "ang_vel_xy",
            "orient",
            "torques",
            "joint_acc",
            "base_height",
            "air_time",
            "collision",
            "stumble",
            "action_rate",
            "hip",
        ]:
            self._episode_sums[key] = torch.zeros(num_envs, device=device)

        self._init_noise_scale_vec(device)

    def _init_height_points(self, device):
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=device)
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        num_envs = self._commands.shape[0] if self._commands is not None else 1
        self._height_points = torch.zeros((num_envs, 140, 3), device=device)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def _init_noise_scale_vec(self, device):
        noise_vec = torch.zeros(188, device=device)
        noise_vec[:3] = self.linear_velocity_noise * self.noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.angular_velocity_noise * self.noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.gravity_noise * self.noise_level
        noise_vec[9:12] = 0.0
        noise_vec[12:24] = self.dof_position_noise * self.noise_level * self.dof_pos_scale
        noise_vec[24:36] = self.dof_velocity_noise * self.noise_level * self.dof_vel_scale
        noise_vec[36:176] = self.height_measurement_noise * self.noise_level * self.height_meas_scale
        noise_vec[176:188] = 0.0
        self._noise_scale_vec = noise_vec

    def set_actions(self, actions):
        self._actions = actions

    def update_buffers(self, dof_vel):
        if self._last_dof_vel is not None:
            self._last_dof_vel = dof_vel.clone()
        if self._last_actions is not None:
            self._last_actions = self._actions.clone()
