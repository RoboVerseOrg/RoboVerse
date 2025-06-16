"""OGBench environment integration for RoboVerse."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from metasim.types import Action, Obs


class OGBenchEnv:
    """Environment wrapper for OGBench tasks."""

    def __init__(self, scenario, handler=None):
        """Initialize OGBench environment.

        Args:
            scenario: The scenario configuration
            handler: The simulator handler (not used for OGBench)
        """
        self.scenario = scenario
        self.task = scenario.task
        self.num_envs = scenario.num_envs

        # Create a minimal handler-like object for compatibility
        class MinimalHandler:
            def __init__(self, scenario):
                self.scenario = scenario
                self.num_envs = scenario.num_envs
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            def get_states(self):
                # Return None to trigger fallback behavior in RLEnvWrapper
                return None

        self.handler = MinimalHandler(scenario)

        # Get the OGBench wrapper
        self._wrapper = self.task.get_wrapper(num_envs=self.num_envs, headless=scenario.headless)

        # Set up spaces
        self.observation_space = self._wrapper.observation_space
        self.action_space = self._wrapper.action_space

        # Initialize episode tracking
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self._max_episode_length = self.task.episode_length

    def reset(
        self,
        env_ids: list[int] | None = None,
        states: list[dict[str, Any]] | None = None,
    ) -> tuple[Obs, dict[str, Any]]:
        """Reset environments.

        Args:
            env_ids: Environment IDs to reset
            states: Initial states (not used for OGBench)

        Returns:
            Observations and info dict
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Reset episode lengths
        for env_id in env_ids:
            self._episode_length_buf[env_id] = 0

        # Reset environment
        obs, info = self._wrapper.reset(env_ids)

        # Convert to list format expected by RoboVerse
        obs_list = []
        for i in range(self.num_envs):
            obs_dict = {
                "observations": obs[i].numpy() if isinstance(obs, torch.Tensor) else obs,
            }

            # Add goal if available
            if "goal" in info:
                obs_dict["goal"] = info["goal"]

            obs_list.append(obs_dict)

        return obs_list, info

    def step(self, action_dict: Action) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the environment.

        Args:
            action_dict: Dictionary of actions

        Returns:
            Tuple of (observations, rewards, successes, timeouts, terminations)
        """
        # Convert action dict to tensor
        if isinstance(action_dict, list):
            # Extract actions from dict format
            actions = []
            for env_action in action_dict:
                if isinstance(env_action, dict):
                    # Assuming flat action space
                    action = next(iter(env_action.values()))
                    if isinstance(action, dict):
                        action = next(iter(action.values()))
                    actions.append(action)
                else:
                    actions.append(env_action)
            actions = torch.tensor(actions, dtype=torch.float32)
        else:
            actions = action_dict

        # Step environment
        obs, rewards, dones, timeouts, info = self._wrapper.step(actions)

        # Update episode lengths
        self._episode_length_buf += 1

        # Check for success
        if "success" in info:
            success = torch.tensor([info["success"]], dtype=torch.bool)
        else:
            success = dones.clone()

        # Convert observations to list format
        obs_list = []
        for i in range(self.num_envs):
            obs_dict = {
                "observations": obs[i].numpy() if isinstance(obs, torch.Tensor) else obs,
            }

            # Add goal if available
            if "goal" in info:
                obs_dict["goal"] = info["goal"]

            obs_list.append(obs_dict)

        # Ensure all tensors have correct shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if success.dim() == 1:
            success = success.unsqueeze(-1)
        if timeouts.dim() == 1:
            timeouts = timeouts.unsqueeze(-1)
        if dones.dim() == 1:
            terminations = dones.unsqueeze(-1)
        else:
            terminations = dones

        return obs_list, rewards, success, timeouts, terminations

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        return self._wrapper.render()

    def close(self):
        """Close the environment."""
        self._wrapper.close()
