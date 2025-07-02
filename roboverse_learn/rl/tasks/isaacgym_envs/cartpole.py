"""Cartpole task wrapper for IsaacGymEnvs.

This wrapper encapsulates the task-specific logic for the Cartpole
balancing task, providing cleaner separation between the
task implementation and the RL infrastructure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    pass

from roboverse_learn.rl.task_registry import register_task_wrapper
from roboverse_learn.rl.task_wrapper import IsaacGymEnvsTaskWrapper

log = logging.getLogger(__name__)


@register_task_wrapper("isaacgym_envs:Cartpole")
class CartpoleTaskWrapper(IsaacGymEnvsTaskWrapper):
    """Task wrapper for Cartpole balancing.

    This task involves balancing a pole on a moving cart by applying
    horizontal forces to the cart.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        super().__init__(env, cfg, sim_type)

        # Task-specific parameters from config
        self.reset_dist = getattr(cfg, "reset_dist", 3.0)
        self.max_push_effort = getattr(cfg, "max_push_effort", 400.0)

        # Observation and action dimensions
        self.num_obs = 4  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        self.num_actions = 1  # Single action for cart force

        # Initialize buffers
        self.initialize_buffers()

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space for Cartpole task."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """Build action space for Cartpole control."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def get_observation_isaacgym(self, states) -> torch.Tensor:
        """Extract observations from IsaacGym states."""
        tensor_state = self._get_tensor_state(states)

        # Get robot state
        robot_state = tensor_state.robots.get("cartpole")
        if robot_state is None:
            log.error("Missing cartpole robot in state")
            return self.obs_buf

        # Get batch size from actual tensor
        batch_size = robot_state.joint_pos.shape[0]

        # Extract joint positions and velocities
        cart_pos = robot_state.joint_pos[:, 0]
        cart_vel = robot_state.joint_vel[:, 0]
        pole_angle = robot_state.joint_pos[:, 1]
        pole_vel = robot_state.joint_vel[:, 1]

        # Assemble observation
        obs = torch.stack(
            [
                cart_pos,  # cart position
                cart_vel,  # cart velocity
                pole_angle,  # pole angle
                pole_vel,  # pole angular velocity
            ],
            dim=-1,
        )

        self.obs_buf[:] = obs
        return self.obs_buf

    def get_observation_mujoco(self, states: list[dict]) -> np.ndarray:
        """Extract observations from MuJoCo states."""
        observations = []

        for env_state in states:
            robot_state = env_state.get("robots", {}).get("cartpole", {})

            # Get joint positions and velocities
            if "joint_qpos" in robot_state:
                joint_pos = robot_state["joint_qpos"]
            elif "dof_pos" in robot_state:
                # Convert dict to array in correct order
                joint_pos = [
                    robot_state["dof_pos"].get("slider_to_cart", 0.0),
                    robot_state["dof_pos"].get("cart_to_pole", 0.0),
                ]
            else:
                joint_pos = [0.0, 0.0]

            if "joint_qvel" in robot_state:
                joint_vel = robot_state["joint_qvel"]
            elif "dof_vel" in robot_state:
                # Convert dict to array in correct order
                joint_vel = [
                    robot_state["dof_vel"].get("slider_to_cart", 0.0),
                    robot_state["dof_vel"].get("cart_to_pole", 0.0),
                ]
            else:
                joint_vel = [0.0, 0.0]

            # Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
            obs = np.array(
                [
                    joint_pos[0],  # cart position
                    joint_vel[0],  # cart velocity
                    joint_pos[1],  # pole angle
                    joint_vel[1],  # pole angular velocity
                ],
                dtype=np.float32,
            )

            observations.append(obs)

        return np.array(observations) if observations else np.zeros((0, self.num_obs))

    def compute_reward_isaacgym(self, states, actions, next_states) -> torch.Tensor:
        """Compute rewards for IsaacGym states."""
        tensor_state = self._get_tensor_state(next_states)

        # Get robot state
        robot_state = tensor_state.robots.get("cartpole")
        if robot_state is None:
            return self.reward_buf

        # Extract states
        cart_pos = robot_state.joint_pos[:, 0]
        cart_vel = robot_state.joint_vel[:, 0]
        pole_angle = robot_state.joint_pos[:, 1]
        pole_vel = robot_state.joint_vel[:, 1]

        # Compute reward similar to IsaacGymEnvs
        # reward = 1.0 - pole_angle^2 - 0.01 * |cart_vel| - 0.005 * |pole_vel|
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

        # Penalties for exceeding bounds
        reward = torch.where(torch.abs(cart_pos) > self.reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # Ensure reward is 1D
        if reward.dim() > 1:
            reward = reward.squeeze(-1)

        self.reward_buf[:] = reward
        return self.reward_buf

    def compute_reward_mujoco(self, states: list[dict], actions: list[dict], next_states: list[dict]) -> np.ndarray:
        """Compute rewards for MuJoCo states."""
        rewards = []

        for env_state in next_states:
            robot_state = env_state.get("robots", {}).get("cartpole", {})

            if "joint_qpos" in robot_state:
                cart_pos = robot_state["joint_qpos"][0]
                pole_angle = robot_state["joint_qpos"][1]
            elif "dof_pos" in robot_state:
                cart_pos = robot_state["dof_pos"].get("slider_to_cart", 0.0)
                pole_angle = robot_state["dof_pos"].get("cart_to_pole", 0.0)
            else:
                cart_pos = 0.0
                pole_angle = 0.0

            if "joint_qvel" in robot_state:
                cart_vel = robot_state["joint_qvel"][0]
                pole_vel = robot_state["joint_qvel"][1]
            elif "dof_vel" in robot_state:
                cart_vel = robot_state["dof_vel"].get("slider_to_cart", 0.0)
                pole_vel = robot_state["dof_vel"].get("cart_to_pole", 0.0)
            else:
                cart_vel = 0.0
                pole_vel = 0.0

            # Compute reward
            reward = 1.0 - pole_angle**2 - 0.01 * abs(cart_vel) - 0.005 * abs(pole_vel)

            # Penalties
            if abs(cart_pos) > self.reset_dist:
                reward = -2.0
            if abs(pole_angle) > np.pi / 2:
                reward = -2.0

            rewards.append(reward)

        return np.array(rewards)

    def check_termination_isaacgym(self, states) -> torch.Tensor:
        """Check termination conditions for IsaacGym."""
        tensor_state = self._get_tensor_state(states)
        robot_state = tensor_state.robots.get("cartpole")

        if robot_state is None:
            return self.reset_buf

        cart_pos = robot_state.joint_pos[:, 0]
        pole_angle = robot_state.joint_pos[:, 1]

        # Terminate if cart exceeds bounds or pole falls too far
        terminations = (torch.abs(cart_pos) > self.reset_dist) | (torch.abs(pole_angle) > np.pi / 2)

        self.reset_buf[:] = terminations.float()
        return self.reset_buf

    def check_termination_mujoco(self, states: list[dict]) -> list[bool]:
        """Check termination conditions for MuJoCo."""
        terminations = []

        for env_state in states:
            robot_state = env_state.get("robots", {}).get("cartpole", {})

            if "joint_qpos" in robot_state:
                cart_pos = robot_state["joint_qpos"][0]
                pole_angle = robot_state["joint_qpos"][1]
            elif "dof_pos" in robot_state:
                cart_pos = robot_state["dof_pos"].get("slider_to_cart", 0.0)
                pole_angle = robot_state["dof_pos"].get("cart_to_pole", 0.0)
            else:
                cart_pos = 0.0
                pole_angle = 0.0

            # Check termination conditions
            terminated = abs(cart_pos) > self.reset_dist or abs(pole_angle) > np.pi / 2
            terminations.append(terminated)

        return terminations

    def reset_task_isaacgym(self, env_ids: list[int] | None = None):
        """Reset task-specific state for IsaacGym."""
        # No task-specific state to reset for Cartpole
        pass

    def reset_task_mujoco(self, env_ids: list[int] | None = None):
        """Reset task-specific state for MuJoCo."""
        # No task-specific state to reset for Cartpole
        pass

    def _get_observation_generic(self, states) -> np.ndarray | torch.Tensor:
        """Fallback observation extraction."""
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
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.compute_reward_mujoco(states, actions, next_states)
        else:
            return np.ones(len(actions))

    def _check_termination_generic(self, states):
        """Generic termination check."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.check_termination_isaacgym(states)
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.check_termination_mujoco(states)
        else:
            return False

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset."""
        pass
