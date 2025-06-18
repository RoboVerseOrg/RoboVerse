"""Wrapper for OGBench environments."""

from __future__ import annotations

import os

# Add ogbench to Python path if needed
import sys
from typing import Any

import numpy as np
import torch

ogbench_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "ogbench"
)
if ogbench_path not in sys.path:
    sys.path.insert(0, ogbench_path)

import ogbench

# Import OGBench submodules to register environments
import ogbench.locomaze  # For maze navigation tasks
import ogbench.manipspace  # For manipulation tasks


class OGBenchWrapper:
    """Wrapper for OGBench environments to integrate with RoboVerse RL infrastructure."""

    def __init__(
        self,
        dataset_name: str,
        num_envs: int = 1,
        headless: bool = True,
        single_task: bool = False,
        task_id: int | None = None,
        use_oracle_rep: bool = False,
        terminate_at_goal: bool = True,
        add_noise_to_goal: bool = True,
        episode_length: int = 1000,
    ):
        """Initialize OGBench wrapper.

        Args:
            dataset_name: Name of the OGBench dataset/environment
            num_envs: Number of parallel environments (currently only 1 is supported)
            headless: Whether to run without rendering (not used for OGBench)
            single_task: Whether to use single-task version
            task_id: Task ID for single-task mode (1-5)
            use_oracle_rep: Whether to use oracle goal representations
            terminate_at_goal: Whether to terminate when goal is reached
            add_noise_to_goal: Whether to add noise to goal position
            episode_length: Maximum episode length
        """
        self.dataset_name = dataset_name
        self.num_envs = num_envs
        self.headless = headless
        self.single_task = single_task
        self.task_id = task_id
        self.use_oracle_rep = use_oracle_rep
        self.terminate_at_goal = terminate_at_goal
        self.add_noise_to_goal = add_noise_to_goal
        self.episode_length = episode_length

        # OGBench currently doesn't support vectorized environments
        if num_envs > 1:
            raise ValueError("OGBench currently only supports single environment (num_envs=1)")

        # Create environment
        env_kwargs = {
            "terminate_at_goal": terminate_at_goal,
        }

        # Only add noise_to_goal for maze environments (not cube)
        if "maze" in dataset_name.lower():
            env_kwargs["add_noise_to_goal"] = add_noise_to_goal

        if use_oracle_rep:
            env_kwargs["use_oracle_rep"] = True

        # For single-task mode, we need to modify the dataset name
        if single_task and task_id is not None:
            # Add singletask suffix if not already present
            if "singletask" not in dataset_name:
                parts = dataset_name.split("-")
                # Insert 'singletask-task{task_id}' before version
                version = parts[-1]  # e.g., 'v0'
                base_parts = parts[:-1]  # e.g., ['antmaze', 'large', 'navigate']
                dataset_name = "-".join(base_parts + [f"singletask-task{task_id}", version])

        # Set render mode based on headless parameter
        if not headless:
            env_kwargs["render_mode"] = "human"

        # Create environment only (no dataset loading in wrapper)
        self.env = ogbench.make_env_and_datasets(dataset_name, env_only=True, **env_kwargs)

        # Store observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Get unwrapped environment for direct access
        self._unwrapped_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        # Initialize buffers
        self._last_obs = None
        self._last_info = None
        self._episode_length = 0

        # For cube environments, we need to handle visualization differently
        self._is_cube_env = "cube" in dataset_name.lower()
        self._viewer_launched = False

    def reset(self, env_ids: list[int] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset environment(s).

        Args:
            env_ids: Environment indices to reset (not used for single env)

        Returns:
            Observations and info dict
        """
        reset_kwargs = {}

        # For goal-conditioned tasks, we need to set task_id
        if not self.single_task:
            # For evaluation, task_id should be provided externally
            # For training, we can randomly sample
            if self.task_id is not None:
                reset_kwargs["task_id"] = self.task_id
            else:
                # Random task for training
                reset_kwargs["task_id"] = np.random.randint(1, 6)
            reset_kwargs["render_goal"] = False  # We don't need rendered goals for RL

        obs, info = self.env.reset(options=reset_kwargs)

        self._last_obs = obs
        self._last_info = info
        self._episode_length = 0

        # Handle visualization on reset for cube environments
        if not self.headless and self._is_cube_env and not self._viewer_launched:
            if hasattr(self._unwrapped_env, "launch_passive_viewer"):
                # Launch passive viewer for cube environments
                self._unwrapped_env.launch_passive_viewer()
                self._viewer_launched = True
                # Set camera to a good viewing angle
                import mujoco

                if (
                    hasattr(self._unwrapped_env, "_passive_viewer_handle")
                    and self._unwrapped_env._passive_viewer_handle
                ):
                    mujoco.mjv_defaultFreeCamera(
                        self._unwrapped_env._model, self._unwrapped_env._passive_viewer_handle.cam
                    )
                    # Adjust camera position for better view
                    self._unwrapped_env._passive_viewer_handle.cam.distance = 1.5
                    self._unwrapped_env._passive_viewer_handle.cam.elevation = -20
                    self._unwrapped_env._passive_viewer_handle.cam.azimuth = 135

        # Convert to torch tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        return obs_tensor, info

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment.

        Args:
            actions: Actions to take

        Returns:
            Tuple of (observations, rewards, dones, timeouts, info)
        """
        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            if actions.dim() == 2:
                actions = actions[0]  # Remove batch dimension
            actions = actions.cpu().numpy()
        elif isinstance(actions, np.ndarray) and actions.ndim == 2:
            actions = actions[0]  # Remove batch dimension

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(actions)

        self._last_obs = obs
        self._last_info = info
        self._episode_length += 1

        # Handle visualization for cube environments
        if not self.headless and self._is_cube_env:
            if not self._viewer_launched and hasattr(self._unwrapped_env, "launch_passive_viewer"):
                # Launch passive viewer for cube environments
                self._unwrapped_env.launch_passive_viewer()
                self._viewer_launched = True
                # Set camera to a good viewing angle
                import mujoco

                if (
                    hasattr(self._unwrapped_env, "_passive_viewer_handle")
                    and self._unwrapped_env._passive_viewer_handle
                ):
                    mujoco.mjv_defaultFreeCamera(
                        self._unwrapped_env._model, self._unwrapped_env._passive_viewer_handle.cam
                    )
                    # Adjust camera position for better view
                    self._unwrapped_env._passive_viewer_handle.cam.distance = 1.5
                    self._unwrapped_env._passive_viewer_handle.cam.elevation = -20
                    self._unwrapped_env._passive_viewer_handle.cam.azimuth = 135

            # Sync viewer with current state
            if self._viewer_launched and hasattr(self._unwrapped_env, "sync_passive_viewer"):
                self._unwrapped_env.sync_passive_viewer()

        # Check for timeout
        timeout = self._episode_length >= self.episode_length
        if timeout:
            truncated = True

        # Convert to torch tensors with batch dimension
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor([terminated or truncated], dtype=torch.bool).unsqueeze(0)
        timeout_tensor = torch.tensor([truncated], dtype=torch.bool).unsqueeze(0)

        return obs_tensor, reward_tensor, done_tensor, timeout_tensor, info

    def get_observation(self) -> torch.Tensor:
        """Get current observation."""
        if self._last_obs is None:
            return torch.zeros((1, *self.observation_space.shape), dtype=torch.float32)
        return torch.tensor(self._last_obs, dtype=torch.float32).unsqueeze(0)

    def get_goal(self) -> torch.Tensor | None:
        """Get current goal observation for goal-conditioned tasks."""
        if self._last_info is not None and "goal" in self._last_info:
            return torch.tensor(self._last_info["goal"], dtype=torch.float32).unsqueeze(0)
        return None

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        # Close passive viewer if it was launched
        if self._viewer_launched and hasattr(self._unwrapped_env, "close_passive_viewer"):
            self._unwrapped_env.close_passive_viewer()
            self._viewer_launched = False
        self.env.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "env"):
            self.close()
