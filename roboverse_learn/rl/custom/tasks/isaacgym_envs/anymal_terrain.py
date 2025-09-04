"""AnymalTerrain task wrapper for IsaacGymEnvs.

This wrapper encapsulates the task-specific logic for the Anymal
terrain locomotion task, providing cleaner separation between the
task implementation and the RL infrastructure.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from gymnasium import spaces

from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.task_registry import register_task_wrapper
from roboverse_learn.rl.task_wrapper import IsaacGymEnvsTaskWrapper

log = logging.getLogger(__name__)


@register_task_wrapper("isaacgym_envs:AnymalTerrain")
class AnymalTerrainTaskWrapper(IsaacGymEnvsTaskWrapper):
    """Task wrapper for Anymal terrain locomotion.

    This task involves controlling a quadruped robot to navigate
    various terrains while following velocity commands.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        super().__init__(env, cfg, sim_type)

        # Task-specific parameters from config
        self.terrain_type = getattr(cfg, "terrain_type", "plane")
        self.terrain_curriculum = getattr(cfg, "terrain_curriculum", True)
        self.terrain_num_levels = getattr(cfg, "terrain_num_levels", 5)
        self.terrain_num_terrains = getattr(cfg, "terrain_num_terrains", 8)

        self.lin_vel_scale = getattr(cfg, "lin_vel_scale", 2.0)
        self.ang_vel_scale = getattr(cfg, "ang_vel_scale", 0.25)
        self.dof_pos_scale = getattr(cfg, "dof_pos_scale", 1.0)
        self.dof_vel_scale = getattr(cfg, "dof_vel_scale", 0.05)
        self.height_meas_scale = getattr(cfg, "height_meas_scale", 5.0)
        self.action_scale = getattr(cfg, "action_scale", 0.5)

        # Reward scales
        self.terminal_reward = getattr(cfg, "terminal_reward", -0.0)
        self.lin_vel_xy_reward_scale = getattr(cfg, "lin_vel_xy_reward_scale", 1.0)
        self.lin_vel_z_reward_scale = getattr(cfg, "lin_vel_z_reward_scale", -4.0)
        self.ang_vel_z_reward_scale = getattr(cfg, "ang_vel_z_reward_scale", 0.5)
        self.ang_vel_xy_reward_scale = getattr(cfg, "ang_vel_xy_reward_scale", -0.05)
        self.orient_reward_scale = getattr(cfg, "orient_reward_scale", -0.0)
        self.torque_reward_scale = getattr(cfg, "torque_reward_scale", -0.00001)
        self.joint_acc_reward_scale = getattr(cfg, "joint_acc_reward_scale", -0.0005)
        self.base_height_reward_scale = getattr(cfg, "base_height_reward_scale", -0.0)
        self.feet_air_time_reward_scale = getattr(cfg, "feet_air_time_reward_scale", 1.0)
        self.knee_collision_reward_scale = getattr(cfg, "knee_collision_reward_scale", -0.25)
        self.feet_stumble_reward_scale = getattr(cfg, "feet_stumble_reward_scale", -0.0)
        self.action_rate_reward_scale = getattr(cfg, "action_rate_reward_scale", -0.01)
        self.hip_reward_scale = getattr(cfg, "hip_reward_scale", -0.0)

        # Command ranges
        self.command_x_range = getattr(cfg, "command_x_range", [-1.0, 1.0])
        self.command_y_range = getattr(cfg, "command_y_range", [-1.0, 1.0])
        self.command_yaw_range = getattr(cfg, "command_yaw_range", [-1.0, 1.0])

        self.allow_knee_contacts = getattr(cfg, "allow_knee_contacts", False)

        # Scale rewards by dt (from post_init in config)
        self.decimation = getattr(cfg, "decimation", 4)
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

        # Observation and action dimensions
        self.num_obs = 188  # 3 + 3 + 3 + 3 + 12 + 12 + 140 + 12
        self.num_actions = 12  # Anymal has 12 DOFs

        # Initialize buffers
        self.initialize_buffers()

        # Task-specific state
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
        self._terminations = None

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space for AnymalTerrain task."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """Build action space for Anymal control."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def get_observation_isaacgym(self, states) -> torch.Tensor:
        """Extract observations from IsaacGym states."""
        tensor_state = self._get_tensor_state(states)

        # Get robot state
        robot_state = tensor_state.robots.get("anymal")
        if robot_state is None:
            log.error("Missing anymal robot in state")
            return self.obs_buf

        # Get batch size from actual tensor
        batch_size = robot_state.root_state.shape[0]

        # Extract root states
        base_pos = robot_state.root_state[:, 0:3]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = robot_state.root_state[:, 7:10]
        base_ang_vel = robot_state.root_state[:, 10:13]

        # Transform velocities to base frame
        base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel) * self.lin_vel_scale
        base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel) * self.ang_vel_scale

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec.unsqueeze(0).repeat(batch_size, 1))

        # Joint states
        dof_pos = robot_state.joint_pos
        dof_vel = robot_state.joint_vel

        # Scale joint positions and velocities
        dof_pos_scaled = dof_pos * self.dof_pos_scale
        dof_vel_scaled = dof_vel * self.dof_vel_scale

        # Initialize commands if needed
        if self._commands is None or self._commands.shape[0] != batch_size:
            self._init_commands(batch_size)

        # Scale commands
        commands_scaled = self._commands[:, :3] * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float32, device=self.device
        )

        # Initialize measured heights if needed
        if self._measured_heights is None or self._measured_heights.shape[0] != batch_size:
            self._measured_heights = torch.zeros((batch_size, 140), dtype=torch.float32, device=self.device)
            self._init_height_points(batch_size)

        # Scale heights
        heights_scaled = (
            torch.clip(base_pos[:, 2].unsqueeze(1) - 0.5 - self._measured_heights, -1, 1.0) * self.height_meas_scale
        )

        # Get actions
        if self._actions is None or self._actions.shape[0] != batch_size:
            self._actions = torch.zeros((batch_size, self.num_actions), dtype=torch.float32, device=self.device)

        # Assemble observation
        obs = torch.cat(
            [
                base_lin_vel_base,  # 3
                base_ang_vel_base,  # 3
                projected_gravity,  # 3
                commands_scaled,  # 3
                dof_pos_scaled,  # 12
                dof_vel_scaled,  # 12
                heights_scaled,  # 140
                self._actions,  # 12
            ],
            dim=-1,
        )

        self.obs_buf[:] = obs
        return self.obs_buf

    def compute_reward_isaacgym(self, states, actions, next_states) -> torch.Tensor:
        """Compute rewards for IsaacGym states."""
        tensor_state = self._get_tensor_state(next_states)

        # Get robot state
        robot_state = tensor_state.robots.get("anymal")
        if robot_state is None:
            return self.reward_buf

        # Extract states
        base_pos = robot_state.root_state[:, 0:3]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = robot_state.root_state[:, 7:10]
        base_ang_vel = robot_state.root_state[:, 10:13]
        dof_pos = robot_state.joint_pos
        dof_vel = robot_state.joint_vel

        batch_size = base_pos.shape[0]

        # Transform velocities to base frame
        base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel)

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=base_pos.device).repeat(batch_size, 1)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Get commands
        if self._commands is None:
            self._init_commands(batch_size)
        commands = self._commands

        # Velocity tracking rewards
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel_base[:, :2]), dim=1)
        ang_vel_error = torch.square(commands[:, 2] - base_ang_vel_base[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.lin_vel_xy_reward_scale
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.ang_vel_z_reward_scale

        # Other velocity penalties
        rew_lin_vel_z = torch.square(base_lin_vel_base[:, 2]) * self.lin_vel_z_reward_scale
        rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel_base[:, :2]), dim=1) * self.ang_vel_xy_reward_scale

        # Orientation penalty
        rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.orient_reward_scale

        # Base height penalty
        rew_base_height = torch.square(base_pos[:, 2] - 0.52) * self.base_height_reward_scale

        # Torque penalty
        if hasattr(robot_state, "torques"):
            torques = robot_state.torques
        else:
            torques = torch.zeros((batch_size, 12), device=base_pos.device)
        rew_torque = torch.sum(torch.square(torques), dim=1) * self.torque_reward_scale

        # Joint acceleration penalty
        if self._last_dof_vel is not None:
            rew_joint_acc = torch.sum(torch.square(self._last_dof_vel - dof_vel), dim=1) * self.joint_acc_reward_scale
        else:
            rew_joint_acc = torch.zeros(batch_size, device=base_pos.device)

        # Contact-based rewards
        if hasattr(robot_state, "contact_forces"):
            contact_forces = robot_state.contact_forces

            # Knee collision penalty (disabled for now)
            knee_contact = torch.zeros(batch_size, dtype=torch.bool, device=base_pos.device)
            rew_collision = knee_contact.float() * self.knee_collision_reward_scale

            # Feet air time reward
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
                self._feet_air_time = torch.zeros((batch_size, 4), device=base_pos.device)
                rew_air_time = torch.zeros(batch_size, device=base_pos.device)

            # Stumble penalty
            stumble = (torch.norm(contact_forces[:, -4:, :2], dim=2) > 5.0) * (
                torch.abs(contact_forces[:, -4:, 2]) < 1.0
            )
            rew_stumble = torch.sum(stumble, dim=1) * self.feet_stumble_reward_scale
        else:
            rew_collision = torch.zeros(batch_size, device=base_pos.device)
            rew_air_time = torch.zeros(batch_size, device=base_pos.device)
            rew_stumble = torch.zeros(batch_size, device=base_pos.device)

        # Action rate penalty
        if self._last_actions is not None:
            rew_action_rate = (
                torch.sum(torch.square(self._last_actions - self._actions), dim=1) * self.action_rate_reward_scale
            )
        else:
            rew_action_rate = torch.zeros(batch_size, device=base_pos.device)

        # Hip penalty
        hip_indices = [0, 3, 6, 9]
        default_hip_pos = torch.tensor([0.03, 0.03, -0.03, -0.03], device=dof_pos.device, dtype=dof_pos.dtype)
        rew_hip = torch.sum(torch.abs(dof_pos[:, hip_indices] - default_hip_pos), dim=1) * self.hip_reward_scale

        # Total reward
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

        # Terminal reward
        if self._terminations is not None:
            total_reward += self.terminal_reward * self._terminations.float()

        # Update episode sums if tracking
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

        # Update actions and velocities
        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
            actions_tensor = torch.zeros((batch_size, self.num_actions), device=base_pos.device)
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        actions_tensor[i] = torch.tensor(action_values, device=base_pos.device)
            self._actions = actions_tensor
        elif isinstance(actions, torch.Tensor):
            self._actions = actions.to(base_pos.device)

        # Update buffers
        if self._last_dof_vel is None:
            self._last_dof_vel = torch.zeros_like(dof_vel)
        else:
            self._last_dof_vel = dof_vel.clone()

        if self._last_actions is None:
            self._last_actions = torch.zeros_like(self._actions)
        else:
            self._last_actions = self._actions.clone()

        self.reward_buf[:] = total_reward
        return self.reward_buf

    def check_termination_isaacgym(self, states) -> torch.Tensor:
        """Check termination conditions for IsaacGym."""
        tensor_state = self._get_tensor_state(states)
        robot_state = tensor_state.robots.get("anymal")

        if robot_state is None:
            return self.reset_buf

        batch_size = robot_state.root_state.shape[0]
        terminations = torch.zeros(batch_size, dtype=torch.bool, device=robot_state.root_state.device)

        # Check for base contact (robot falling)
        if hasattr(robot_state, "contact_forces"):
            contact_forces = robot_state.contact_forces
            base_contact = torch.norm(contact_forces[:, 0, :], dim=1) > 1.0
            terminations |= base_contact

            # Knee contacts if not allowed
            if not self.allow_knee_contacts:
                # Would add knee contact check here if indices were known
                pass

        self._terminations = terminations
        self.reset_buf[:] = terminations.float()
        return self.reset_buf

    def reset_task_isaacgym(self, env_ids: list[int] | None = None):
        """Reset task-specific state for IsaacGym."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num_reset = len(env_ids)

        # Generate new random commands for reset environments
        commands = torch.zeros((num_reset, 4), dtype=torch.float32, device=self.device)
        commands[:, 0] = (
            torch.rand(num_reset, device=self.device) * (self.command_x_range[1] - self.command_x_range[0])
            + self.command_x_range[0]
        )
        commands[:, 1] = (
            torch.rand(num_reset, device=self.device) * (self.command_y_range[1] - self.command_y_range[0])
            + self.command_y_range[0]
        )
        commands[:, 2] = (
            torch.rand(num_reset, device=self.device) * (self.command_yaw_range[1] - self.command_yaw_range[0])
            + self.command_yaw_range[0]
        )
        commands[:, 3] = 0.0

        # Zero out commands for small velocities
        commands *= (torch.norm(commands[:, :2], dim=1) > 0.25).unsqueeze(1)

        # Update commands for reset environments
        if self._commands is not None:
            for i, env_id in enumerate(env_ids):
                if env_id < len(self._commands):
                    self._commands[env_id] = commands[i]

        # Reset other buffers
        if self._last_actions is not None:
            self._last_actions[env_ids] = 0.0
        if self._last_dof_vel is not None:
            self._last_dof_vel[env_ids] = 0.0
        if self._feet_air_time is not None:
            self._feet_air_time[env_ids] = 0.0
        if self._episode_sums is not None:
            for key in self._episode_sums.keys():
                self._episode_sums[key][env_ids] = 0.0

    def _init_commands(self, batch_size: int):
        """Initialize command buffer."""
        self._commands = torch.zeros((batch_size, 4), device=self.device)
        self._commands[:, 0] = (
            torch.rand(batch_size, device=self.device) * (self.command_x_range[1] - self.command_x_range[0])
            + self.command_x_range[0]
        )
        self._commands[:, 1] = (
            torch.rand(batch_size, device=self.device) * (self.command_y_range[1] - self.command_y_range[0])
            + self.command_y_range[0]
        )
        self._commands[:, 2] = (
            torch.rand(batch_size, device=self.device) * (self.command_yaw_range[1] - self.command_yaw_range[0])
            + self.command_yaw_range[0]
        )
        self._commands[:, 3] = 0.0

        # Zero out commands for small velocities
        self._commands *= (torch.norm(self._commands[:, :2], dim=1) > 0.25).unsqueeze(1)

    def _init_height_points(self, batch_size: int):
        """Initialize height measurement points."""
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device)
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self._height_points = torch.zeros((batch_size, 140, 3), device=self.device)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def initialize_buffers(self):
        """Initialize task-specific buffers."""
        super().initialize_buffers()

        # Get actual number of environments
        actual_num_envs = self.num_envs
        if hasattr(self.env, "num_envs"):
            actual_num_envs = self.env.num_envs
        elif hasattr(self.env, "handler") and hasattr(self.env.handler, "num_envs"):
            actual_num_envs = self.env.handler.num_envs

        # Initialize episode sums for tracking
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
            self._episode_sums[key] = torch.zeros(actual_num_envs, device=self.device)

        # Initialize terrain levels if using curriculum
        if self.terrain_curriculum:
            max_init_level = self.terrain_num_levels - 1
        else:
            max_init_level = 0
        self._terrain_levels = torch.randint(0, max_init_level + 1, (actual_num_envs,), device=self.device)
        self._terrain_types = torch.randint(0, self.terrain_num_terrains, (actual_num_envs,), device=self.device)

    def _get_observation_generic(self, states) -> np.ndarray | torch.Tensor:
        """Fallback observation extraction."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.get_observation_isaacgym(states)
        else:
            log.warning(f"Unknown state format: {type(states)}")
            return self.obs_buf if hasattr(self, "obs_buf") else np.zeros((1, self.num_obs))

    def _compute_reward_generic(self, states, actions, next_states):
        """Fallback reward computation."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.compute_reward_isaacgym(states, actions, next_states)
        else:
            return np.ones(len(actions))

    def _check_termination_generic(self, states):
        """Generic termination check."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.check_termination_isaacgym(states)
        else:
            return False

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset."""
        pass
