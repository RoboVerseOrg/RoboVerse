"""Anymal task wrapper for IsaacGymEnvs.

This wrapper encapsulates the task-specific logic for the Anymal
quadruped locomotion task, providing cleaner separation between the
task implementation and the RL infrastructure.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from gymnasium import spaces

from metasim.utils.math import quat_rotate, quat_rotate_inverse
from roboverse_learn.rl.task_registry import register_task_wrapper
from roboverse_learn.rl.task_wrapper import IsaacGymEnvsTaskWrapper

log = logging.getLogger(__name__)


@register_task_wrapper("isaacgym_envs:Anymal")
class AnymalTaskWrapper(IsaacGymEnvsTaskWrapper):
    """Task wrapper for Anymal quadruped locomotion.

    This task involves controlling a quadruped robot to follow
    velocity commands while maintaining stability.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        super().__init__(env, cfg, sim_type)

        # Task-specific parameters from config
        self.lin_vel_scale = getattr(cfg, "lin_vel_scale", 2.0)
        self.ang_vel_scale = getattr(cfg, "ang_vel_scale", 0.25)
        self.dof_pos_scale = getattr(cfg, "dof_pos_scale", 1.0)
        self.dof_vel_scale = getattr(cfg, "dof_vel_scale", 0.05)
        self.action_scale = getattr(cfg, "action_scale", 0.5)

        self.lin_vel_xy_reward_scale = getattr(cfg, "lin_vel_xy_reward_scale", 1.0)
        self.ang_vel_z_reward_scale = getattr(cfg, "ang_vel_z_reward_scale", 0.5)
        self.torque_reward_scale = getattr(cfg, "torque_reward_scale", -0.000025)

        self.command_x_range = getattr(cfg, "command_x_range", [-2.0, 2.0])
        self.command_y_range = getattr(cfg, "command_y_range", [-1.0, 1.0])
        self.command_yaw_range = getattr(cfg, "command_yaw_range", [-1.0, 1.0])

        self.base_contact_force_threshold = getattr(cfg, "base_contact_force_threshold", 1.0)
        self.knee_contact_force_threshold = getattr(cfg, "knee_contact_force_threshold", 1.0)

        # Observation and action dimensions
        self.num_obs = 48  # 3 + 3 + 3 + 3 + 12 + 12 + 12
        self.num_actions = 12  # Anymal has 12 DOFs

        # Initialize buffers
        self.initialize_buffers()

        # Task-specific state
        self._commands = None
        self._prev_actions = None

        # Default joint positions for Anymal
        self.default_dof_pos = torch.tensor(
            [0.03, 0.03, -0.03, -0.03, 0.4, -0.4, 0.4, -0.4, -0.8, 0.8, -0.8, 0.8],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space for Anymal task."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """Build action space for Anymal control."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def get_observation_isaacgym(self, states) -> torch.Tensor:
        """Extract observations from IsaacGym states.

        Handles the TensorState format specific to IsaacGym.
        """
        tensor_state = self._get_tensor_state(states)

        # Get robot state
        robot_state = tensor_state.robots.get("anymal")
        if robot_state is None:
            log.error("Missing anymal robot in state")
            return self.obs_buf

        # Get batch size from actual tensor
        batch_size = robot_state.root_state.shape[0]

        # Extract root states
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = robot_state.root_state[:, 7:10]
        base_ang_vel = robot_state.root_state[:, 10:13]

        # Transform velocities to base frame
        base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel) * self.lin_vel_scale
        base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel) * self.ang_vel_scale

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = quat_rotate(base_quat, gravity_vec.unsqueeze(0).repeat(batch_size, 1))

        # Joint states
        dof_pos = robot_state.joint_pos
        dof_vel = robot_state.joint_vel

        # Scale joint positions relative to default
        dof_pos_scaled = (dof_pos - self.default_dof_pos.unsqueeze(0)) * self.dof_pos_scale
        dof_vel_scaled = dof_vel * self.dof_vel_scale

        # Initialize commands if needed
        if self._commands is None or self._commands.shape[0] != batch_size:
            self._commands = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
            # Generate random commands
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

        # Scale commands
        commands_scaled = self._commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float32, device=self.device
        )

        # Get previous actions
        if self._prev_actions is None or self._prev_actions.shape[0] != batch_size:
            self._prev_actions = torch.zeros((batch_size, self.num_actions), dtype=torch.float32, device=self.device)

        # Assemble observation
        obs = torch.cat(
            [
                base_lin_vel_base,  # 3
                base_ang_vel_base,  # 3
                projected_gravity,  # 3
                commands_scaled,  # 3
                dof_pos_scaled,  # 12
                dof_vel_scaled,  # 12
                self._prev_actions,  # 12
            ],
            dim=-1,
        )

        self.obs_buf[:] = obs
        return self.obs_buf

    def get_observation_mujoco(self, states) -> np.ndarray:
        """Extract observations from MuJoCo states."""
        # Initialize observation array
        obs = np.zeros((len(states), self.num_obs))

        for i, state in enumerate(states):
            # Extract robot state
            robot_state = state.get("robots", {}).get("anymal", {})

            # Get base states
            base_quat = torch.tensor(robot_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
            base_lin_vel = torch.tensor(
                robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32
            )
            base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32)

            # Transform velocities to base frame
            base_lin_vel_base = (
                quat_rotate_inverse(base_quat.unsqueeze(0), base_lin_vel.unsqueeze(0)).squeeze(0) * self.lin_vel_scale
            )
            base_ang_vel_base = (
                quat_rotate_inverse(base_quat.unsqueeze(0), base_ang_vel.unsqueeze(0)).squeeze(0) * self.ang_vel_scale
            )

            # Projected gravity
            gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
            projected_gravity = quat_rotate(base_quat.unsqueeze(0), gravity_vec.unsqueeze(0)).squeeze(0)

            # Get joint states
            if "joint_qpos" in robot_state:
                dof_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32)
            elif "dof_pos" in robot_state:
                dof_pos = torch.tensor(list(robot_state["dof_pos"].values()), dtype=torch.float32)
            else:
                dof_pos = torch.zeros(12, dtype=torch.float32)

            if "joint_qvel" in robot_state:
                dof_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32)
            elif "dof_vel" in robot_state:
                dof_vel = torch.tensor(list(robot_state["dof_vel"].values()), dtype=torch.float32)
            else:
                dof_vel = torch.zeros(12, dtype=torch.float32)

            # Scale joint positions and velocities
            dof_pos_scaled = (dof_pos - self.default_dof_pos.cpu()) * self.dof_pos_scale
            dof_vel_scaled = dof_vel * self.dof_vel_scale

            # Get commands and previous actions
            commands = torch.zeros(3, dtype=torch.float32)
            if self._commands is not None and i < len(self._commands):
                commands = self._commands[i].cpu()

            commands_scaled = commands * torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale])

            prev_actions = torch.zeros(12, dtype=torch.float32)
            if self._prev_actions is not None and i < len(self._prev_actions):
                prev_actions = self._prev_actions[i].cpu()

            # Assemble observation
            obs[i] = torch.cat([
                base_lin_vel_base,
                base_ang_vel_base,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
                prev_actions,
            ]).numpy()

        return obs

    def compute_reward_isaacgym(self, states, actions, next_states) -> torch.Tensor:
        """Compute rewards for IsaacGym states."""
        tensor_state = self._get_tensor_state(next_states)

        # Get robot state
        robot_state = tensor_state.robots.get("anymal")
        if robot_state is None:
            return self.reward_buf

        # Extract states
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = robot_state.root_state[:, 7:10]
        base_ang_vel = robot_state.root_state[:, 10:13]

        # Transform velocities to base frame
        base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)

        # Get commands
        if self._commands is not None:
            commands = self._commands
        else:
            num_envs = base_quat.shape[0]
            commands = torch.zeros((num_envs, 3), device=base_quat.device)

        # Velocity tracking rewards
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])

        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.lin_vel_xy_reward_scale
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.ang_vel_z_reward_scale

        # Torque penalty
        if hasattr(robot_state, "torques"):
            torques = robot_state.torques
        else:
            torques = torch.zeros((base_quat.shape[0], 12), device=base_quat.device)

        rew_torque = torch.sum(torch.square(torques), dim=1) * self.torque_reward_scale

        # Total reward
        rewards = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
        rewards = torch.clamp(rewards, min=0.0)

        # Update previous actions
        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
            actions_tensor = torch.zeros((len(actions), self.num_actions), device=base_quat.device)
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        actions_tensor[i] = torch.tensor(action_values, device=base_quat.device)
            self._prev_actions = actions_tensor
        elif isinstance(actions, torch.Tensor):
            self._prev_actions = actions.to(base_quat.device)

        self.reward_buf[:] = rewards
        return self.reward_buf

    def check_termination_isaacgym(self, states) -> torch.Tensor:
        """Check termination conditions for IsaacGym."""
        tensor_state = self._get_tensor_state(states)
        robot_state = tensor_state.robots.get("anymal")

        if robot_state is None:
            return self.reset_buf

        # Check for base or knee contact (would indicate falling)
        if hasattr(robot_state, "contact_forces"):
            contact_forces = robot_state.contact_forces

            # Base contact (first link)
            base_contact = torch.norm(contact_forces[:, 0, :], dim=1) > self.base_contact_force_threshold

            terminations = base_contact
        else:
            # No contact forces available, no termination
            num_envs = robot_state.root_state.shape[0]
            terminations = torch.zeros(num_envs, dtype=torch.bool, device=robot_state.root_state.device)

        self.reset_buf[:] = terminations.float()
        return self.reset_buf

    def reset_task_isaacgym(self, env_ids: list[int] | None = None):
        """Reset task-specific state for IsaacGym."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate new random commands for reset environments
        num_reset = len(env_ids)
        new_commands = torch.zeros((num_reset, 3), dtype=torch.float32, device=self.device)
        new_commands[:, 0] = (
            torch.rand(num_reset, device=self.device) * (self.command_x_range[1] - self.command_x_range[0])
            + self.command_x_range[0]
        )
        new_commands[:, 1] = (
            torch.rand(num_reset, device=self.device) * (self.command_y_range[1] - self.command_y_range[0])
            + self.command_y_range[0]
        )
        new_commands[:, 2] = (
            torch.rand(num_reset, device=self.device) * (self.command_yaw_range[1] - self.command_yaw_range[0])
            + self.command_yaw_range[0]
        )

        # Update commands for reset environments
        if self._commands is not None:
            for i, env_id in enumerate(env_ids):
                if env_id < len(self._commands):
                    self._commands[env_id] = new_commands[i]

        # Reset previous actions
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0

    def _get_observation_generic(self, states) -> np.ndarray | torch.Tensor:
        """Fallback observation extraction."""
        # Try to determine state format and dispatch accordingly
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.get_observation_isaacgym(states)
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.get_observation_mujoco(states)
        else:
            log.warning(f"Unknown state format: {type(states)}")
            return self.obs_buf if hasattr(self, "obs_buf") else np.zeros((1, self.num_obs))

    def _compute_reward_generic(self, states, actions, next_states):
        """Fallback reward computation."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.compute_reward_isaacgym(states, actions, next_states)
        else:
            # Simple implementation for other simulators
            return np.zeros(len(actions))

    def _check_termination_generic(self, states):
        """Generic termination check."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.check_termination_isaacgym(states)
        else:
            return False

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset."""
        pass
