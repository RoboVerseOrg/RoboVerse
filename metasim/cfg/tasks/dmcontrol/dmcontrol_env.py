"""Environment wrapper for dm_control tasks in RoboVerse."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class DMControlEnv:
    """Environment wrapper for dm_control tasks."""

    def __init__(self, scenario):
        """Initialize the dm_control environment wrapper."""

        self.scenario = scenario
        self.task = scenario.task
        self.num_envs = scenario.num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the dm_control wrapper from the task
        self._wrapper = self.task.get_wrapper(num_envs=self.num_envs, headless=scenario.headless)

        # Episode tracking
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Initialize environments
        self._obs = self._wrapper.reset()

        # Launch viewer if not headless
        if not scenario.headless:
            self._wrapper.launch_viewer()

    def reset(
        self, states: list[Any] | None = None, env_ids: list[int] | None = None
    ) -> tuple[dict[str, torch.Tensor], None]:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Reset episode counters
        self._episode_length_buf[env_ids] = 0

        # Reset environments in wrapper
        self._obs = self._wrapper.reset(env_ids=env_ids)

        # Return raw observations - RLEnvWrapper will process them
        return self._obs, None

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Step all environments with given actions."""
        # Increment episode counters
        self._episode_length_buf += 1

        # Step the wrapper
        obs, rewards, dones, _ = self._wrapper.step(actions)

        # Check timeouts
        timeouts = self._episode_length_buf >= self.scenario.episode_length

        # Ensure all tensors are on the correct device
        if not isinstance(timeouts, torch.Tensor):
            timeouts = torch.tensor(timeouts, dtype=torch.bool, device=self.device)
        elif timeouts.device != self.device:
            timeouts = timeouts.to(self.device)

        # Reset environments that are done
        done_envs = torch.where(dones | timeouts)[0]
        if len(done_envs) > 0:
            self.reset(env_ids=done_envs.tolist())

        # Return raw data - RLEnvWrapper will process them
        return obs, rewards, dones, timeouts, None

    def render(self) -> None:
        """Render the environment."""
        self._wrapper.render()

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self._wrapper, "close"):
            self._wrapper.close()

    @property
    def episode_length_buf(self) -> list[int]:
        """Get episode length buffer."""
        return self._episode_length_buf.tolist()

    @property
    def observation_space(self):
        """Get observation space."""
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self.task.observation_space["shape"][0]),), dtype=np.float32
        )

    @property
    def action_space(self):
        """Get action space."""
        # Get action bounds from wrapper
        action_spec = self._wrapper.action_spec

        if hasattr(action_spec, "minimum") and hasattr(action_spec, "maximum"):
            return gym.spaces.Box(
                low=action_spec.minimum, high=action_spec.maximum, shape=action_spec.shape, dtype=np.float32
            )
        else:
            # Default action space
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.task.action_dim,), dtype=np.float32)

    # Dummy handler property for compatibility
    @property
    def handler(self):
        """Return self as handler for compatibility."""
        return self

    @property
    def robot(self):
        """No robot for dm_control tasks."""
        return None

    @property
    def robots(self):
        """No robots for dm_control tasks."""
        return []

    def get_states(self):
        """Get current states - returns None for dm_control tasks."""
        # dm_control handles states internally
        return None
