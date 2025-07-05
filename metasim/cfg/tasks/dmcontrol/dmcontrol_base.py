from __future__ import annotations

from typing import Any

import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg
from .dm_wrapper import DM_CONTROL_AVAILABLE, DMControlWrapper


@configclass
class DMControlBaseCfg(BaseTaskCfg):
    """Base configuration for dm_control tasks."""

    episode_length = 1000
    traj_filepath = None
    task_type = TaskType.LOCOMOTION
    objects = []
    checker = EmptyChecker()

    # dm_control specific parameters
    domain_name: str = None
    task_name: str = None
    visualize_reward: bool = False
    time_limit: float | None = None
    environment_kwargs: dict[str, Any] | None = None
    random_state: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self._wrapper = None
        self._initialized = False

        # Set observation space based on dm_control environment
        if DM_CONTROL_AVAILABLE and self.domain_name and self.task_name:
            # Create temporary wrapper to get dimensions
            temp_wrapper = DMControlWrapper(
                domain_name=self.domain_name,
                task_name=self.task_name,
                num_envs=1,
                time_limit=self.time_limit,
                visualize_reward=False,
            )

            self.observation_space = {"shape": [temp_wrapper.obs_dim]}
            self.action_dim = temp_wrapper.action_dim

            temp_wrapper.close()
        else:
            # Default values if dm_control is not available
            self.observation_space = {"shape": [50]}  # Default observation size
            self.action_dim = 6  # Default action size

    def get_wrapper(self, num_envs: int = 1, headless: bool = True) -> DMControlWrapper:
        """Get the dm_control wrapper."""
        if not DM_CONTROL_AVAILABLE:
            raise ImportError("dm_control is not installed. Please install it with: pip install dm-control")

        if self._wrapper is None:
            self._wrapper = DMControlWrapper(
                domain_name=self.domain_name,
                task_name=self.task_name,
                num_envs=num_envs,
                time_limit=self.time_limit,
                visualize_reward=self.visualize_reward,
                environment_kwargs=self.environment_kwargs,
                random_state=self.random_state,
                headless=headless,
            )

        return self._wrapper

    def get_observation(self, obs_tensor):
        """Return observations from dm_control wrapper."""
        # The wrapper already returns flattened torch tensors
        if isinstance(obs_tensor, torch.Tensor):
            return obs_tensor
        else:
            # Convert to tensor if needed
            return torch.tensor(obs_tensor, dtype=torch.float32)

    def reward_fn(self, reward_tensor):
        """Return rewards from dm_control wrapper."""
        # The wrapper already returns reward tensors
        if isinstance(reward_tensor, torch.Tensor):
            return reward_tensor
        else:
            return torch.tensor(reward_tensor, dtype=torch.float32)

    def termination_fn(self, done_tensor):
        """Return terminations from dm_control wrapper."""
        # The wrapper already returns done tensors
        if isinstance(done_tensor, torch.Tensor):
            return done_tensor
        else:
            return torch.tensor(done_tensor, dtype=torch.bool)

    def build_scene(self, config=None):
        """Initialize the dm_control wrapper."""
        self._initialized = True

    def reset(self, env_ids=None):
        """Reset dm_control environment."""
        # Reset handled by wrapper
        pass

    def post_reset(self):
        """Called after reset."""
        pass
