"""AllegroHand task wrapper for IsaacGymEnvs.

This wrapper encapsulates the task-specific logic for the AllegroHand
object reorientation task, providing cleaner separation between the
task implementation and the RL infrastructure.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from gymnasium import spaces

from metasim.utils.math import quat_inv, quat_mul
from roboverse_learn.rl.task_registry import register_task_wrapper
from roboverse_learn.rl.task_wrapper import IsaacGymEnvsTaskWrapper

log = logging.getLogger(__name__)


@register_task_wrapper("isaacgym_envs:AllegroHand")
class AllegroHandTaskWrapper(IsaacGymEnvsTaskWrapper):
    """Task wrapper for AllegroHand object reorientation.

    This task involves controlling an Allegro hand to reorient a block
    to match a target orientation.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        super().__init__(env, cfg, sim_type)

        # Task-specific parameters from config
        self.obs_type = getattr(cfg, "obs_type", "full_no_vel")
        self.dist_reward_scale = getattr(cfg, "dist_reward_scale", -10.0)
        self.rot_reward_scale = getattr(cfg, "rot_reward_scale", 1.0)
        self.action_penalty_scale = getattr(cfg, "action_penalty_scale", -0.0002)
        self.reach_goal_bonus = getattr(cfg, "reach_goal_bonus", 250.0)
        self.success_tolerance = getattr(cfg, "success_tolerance", 0.1)
        self.rot_eps = getattr(cfg, "rot_eps", 0.1)
        self.av_factor = getattr(cfg, "av_factor", 0.1)

        # Observation dimensions based on obs_type
        self.obs_dims = {"full_no_vel": 50, "full": 72, "full_state": 88}
        self.num_obs = self.obs_dims.get(self.obs_type, 50)
        self.num_actions = 16  # Allegro hand has 16 DoFs

        # Initialize buffers
        self.initialize_buffers()

        # Task-specific state
        self._prev_actions = None
        self._consecutive_successes = None

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space based on obs_type."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """Build action space for Allegro hand control."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def get_observation_isaacgym(self, states) -> torch.Tensor:
        """Extract observations from IsaacGym states.

        Handles the TensorState format specific to IsaacGym.
        """
        tensor_state = self._get_tensor_state(states)

        # Get object states
        object_state = tensor_state.objects.get("block")
        goal_state = tensor_state.objects.get("goal")
        robot_state = tensor_state.robots.get("allegro_hand")

        if object_state is None or goal_state is None or robot_state is None:
            log.error("Missing required objects/robots in state")
            return self.obs_buf

        # Extract positions and rotations
        object_pos = object_state.root_state[:, :3]
        object_rot = object_state.root_state[:, 3:7]
        goal_pos = goal_state.root_state[:, :3]
        goal_rot = goal_state.root_state[:, 3:7]

        # Hand joint positions
        hand_pos = robot_state.joint_pos

        # Compute quaternion difference
        quat_diff = quat_mul(object_rot, quat_inv(goal_rot))

        # Ensure all tensors are on the same device
        device = hand_pos.device

        if self.obs_type == "full_no_vel":
            # Get batch size from actual tensor
            batch_size = hand_pos.shape[0]

            # Ensure prev_actions has correct shape
            prev_actions = self._get_prev_actions()
            if prev_actions.shape[0] != batch_size:
                # Create prev_actions with correct batch size
                prev_actions = torch.zeros((batch_size, self.num_actions), dtype=torch.float32, device=device)
                self._prev_actions = prev_actions
            else:
                prev_actions = prev_actions.to(device)

            # Basic observation without velocities
            obs = torch.cat(
                [
                    hand_pos,  # 16
                    object_pos,  # 3
                    object_rot,  # 4
                    goal_pos,  # 3
                    goal_rot,  # 4
                    quat_diff,  # 4
                    prev_actions,  # 16
                ],
                dim=-1,
            )

        elif self.obs_type == "full":
            # Get batch size from actual tensor
            batch_size = hand_pos.shape[0]

            # Ensure prev_actions has correct shape
            prev_actions = self._get_prev_actions()
            if prev_actions.shape[0] != batch_size:
                prev_actions = torch.zeros((batch_size, self.num_actions), dtype=torch.float32, device=device)
                self._prev_actions = prev_actions
            else:
                prev_actions = prev_actions.to(device)

            # Full observation with velocities
            hand_vel = robot_state.joint_vel * 0.2
            object_vel = object_state.root_state[:, 7:10]
            object_ang_vel = object_state.root_state[:, 10:13] * 0.2

            obs = torch.cat(
                [
                    hand_pos,  # 16
                    hand_vel,  # 16
                    torch.zeros_like(hand_pos),  # 16 (placeholder for hand forces)
                    object_pos,  # 3
                    object_rot,  # 4
                    object_vel,  # 3
                    object_ang_vel,  # 3
                    goal_pos,  # 3
                    goal_rot,  # 4
                    quat_diff,  # 4
                    prev_actions,  # 16
                ],
                dim=-1,
            )

        else:
            # Default to full_no_vel
            obs = self.get_observation_isaacgym(states)

        self.obs_buf[:] = obs
        return self.obs_buf

    def get_observation_mujoco(self, states) -> np.ndarray:
        """Extract observations from MuJoCo states.

        Handles dictionary-based state format.
        """
        # Initialize observation array
        obs = np.zeros((len(states), self.num_obs))

        for i, state in enumerate(states):
            # Extract robot state
            robot_state = state.get("robot", {}).get("allegro_hand", {})
            hand_pos = np.array([robot_state.get("joint_qpos", {}).get(f"joint_{j}", 0.0) for j in range(16)])

            # Extract object states
            block_state = state.get("object", {}).get("block", {})
            goal_state = state.get("object", {}).get("goal", {})

            object_pos = np.array(block_state.get("position", [0.0, 0.0, 0.0]))
            object_rot = np.array(block_state.get("orientation", [1.0, 0.0, 0.0, 0.0]))
            goal_pos = np.array(goal_state.get("position", [0.0, 0.0, 0.0]))
            goal_rot = np.array(goal_state.get("orientation", [1.0, 0.0, 0.0, 0.0]))

            # Compute quaternion difference
            quat_diff = quat_mul(
                torch.tensor(object_rot, dtype=torch.float32), quat_inv(torch.tensor(goal_rot, dtype=torch.float32))
            ).numpy()

            # Get previous actions
            prev_actions = self._get_prev_actions_numpy(i)

            # Assemble observation
            obs[i] = np.concatenate([hand_pos, object_pos, object_rot, goal_pos, goal_rot, quat_diff, prev_actions])

        return obs

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

    def compute_reward_isaacgym(self, states, actions, next_states) -> torch.Tensor:
        """Compute rewards for IsaacGym states."""
        tensor_state = self._get_tensor_state(next_states)

        # Get object states
        object_state = tensor_state.objects.get("block")
        goal_state = tensor_state.objects.get("goal")

        if object_state is None or goal_state is None:
            return self.reward_buf

        # Extract positions and rotations
        object_pos = object_state.root_state[:, :3]
        object_rot = object_state.root_state[:, 3:7]
        goal_pos = goal_state.root_state[:, :3]
        goal_rot = goal_state.root_state[:, 3:7]

        # Distance reward
        pos_dist = torch.norm(object_pos - goal_pos, p=2, dim=-1)
        dist_reward = self.dist_reward_scale * pos_dist

        # Rotation reward
        object_rot = torch.nn.functional.normalize(object_rot, p=2, dim=-1)
        goal_rot = torch.nn.functional.normalize(goal_rot, p=2, dim=-1)
        quat_dot = torch.abs(torch.sum(object_rot * goal_rot, dim=-1))
        quat_dot = torch.clamp(quat_dot, min=0.0, max=1.0)
        rot_reward = self.rot_reward_scale * quat_dot

        # Action penalty
        # Convert actions to tensor if it's a list of dicts
        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
            action_tensor = torch.zeros((len(actions), self.num_actions), device=object_pos.device)
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        action_tensor[i] = torch.tensor(action_values, device=object_pos.device)
        else:
            action_tensor = actions

        action_penalty = self.action_penalty_scale * torch.sum(action_tensor**2, dim=-1)

        # Total reward
        reward = dist_reward + rot_reward + action_penalty

        # Success bonus
        success = (pos_dist < self.success_tolerance) & (quat_dot > 1.0 - self.rot_eps)
        reward = torch.where(success, reward + self.reach_goal_bonus, reward)

        # Update consecutive successes
        if self._consecutive_successes is None:
            self._consecutive_successes = torch.zeros(reward.shape[0], dtype=torch.int32, device=reward.device)
        self._consecutive_successes = torch.where(
            success, self._consecutive_successes + 1, torch.zeros_like(self._consecutive_successes)
        )

        self.reward_buf[:] = reward
        return self.reward_buf

    def _compute_reward_generic(self, states, actions, next_states):
        """Fallback reward computation."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.compute_reward_isaacgym(states, actions, next_states)
        else:
            # Simple implementation for other simulators
            return np.zeros(len(actions))

    def check_termination_isaacgym(self, states) -> torch.Tensor:
        """Check termination conditions for IsaacGym."""
        # No early termination in this task
        return self.reset_buf

    def _check_termination_generic(self, states):
        """Generic termination check."""
        return False

    def reset_task_isaacgym(self, env_ids: list[int] | None = None):
        """Reset task-specific state for IsaacGym."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset consecutive successes
        if self._consecutive_successes is not None:
            self._consecutive_successes[env_ids] = 0

        # Reset previous actions
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset."""
        pass

    def _get_prev_actions(self) -> torch.Tensor:
        """Get previous actions tensor."""
        if self._prev_actions is None:
            # Get the actual number of environments from the environment if our cached value is wrong
            actual_num_envs = self.num_envs
            if hasattr(self.env, "num_envs"):
                actual_num_envs = self.env.num_envs
            elif hasattr(self.env, "handler") and hasattr(self.env.handler, "num_envs"):
                actual_num_envs = self.env.handler.num_envs

            self._prev_actions = torch.zeros(
                (actual_num_envs, self.num_actions), dtype=torch.float32, device=self.device
            )
        return self._prev_actions

    def _get_prev_actions_numpy(self, idx: int) -> np.ndarray:
        """Get previous actions for a specific environment."""
        if self._prev_actions is None:
            return np.zeros(self.num_actions, dtype=np.float32)
        elif isinstance(self._prev_actions, torch.Tensor):
            return self._prev_actions[idx].cpu().numpy()
        else:
            return self._prev_actions[idx]

    def update_prev_actions(self, actions):
        """Update previous actions buffer."""
        if isinstance(actions, torch.Tensor):
            if self._prev_actions is None:
                self._prev_actions = actions.clone()
            else:
                self._prev_actions[:] = actions
        elif isinstance(actions, np.ndarray):
            self._prev_actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        elif isinstance(actions, list):
            # Handle list of action dictionaries
            action_tensor = torch.zeros((len(actions), self.num_actions), dtype=torch.float32, device=self.device)
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        action_tensor[i] = torch.tensor(action_values, device=self.device)
            self._prev_actions = action_tensor
