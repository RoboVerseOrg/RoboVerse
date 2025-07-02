"""Task-specific wrapper layer for RL environments.

This module provides a wrapper layer between RLEnvWrapper and GymEnvWrapper that
encapsulates all task-specific logic (observations, rewards, resets, terminations).
It supports simulator-specific implementations via method dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    pass
from gymnasium import spaces

from metasim.utils.state import TensorState


class BaseTaskWrapper(ABC):
    """Abstract base class for task-specific wrappers.

    This class defines the interface that all task wrappers must implement.
    It provides methods for observation extraction, reward computation,
    termination checking, and task-specific resets.
    """

    def __init__(self, env, cfg, sim_type: str):
        """Initialize the task wrapper.

        Args:
            env: The wrapped environment (GymEnvWrapper instance)
            cfg: Task configuration object
            sim_type: Simulator type (e.g., 'isaacgym', 'mujoco', 'sapien')
        """
        self.env = env
        self.cfg = cfg
        self.sim_type = sim_type
        self._observation_space = None
        self._action_space = None

    @property
    def observation_space(self):
        """Get the observation space for this task."""
        if self._observation_space is None:
            self._observation_space = self._build_observation_space()
        return self._observation_space

    @property
    def action_space(self):
        """Get the action space for this task."""
        if self._action_space is None:
            self._action_space = self._build_action_space()
        return self._action_space

    @abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """Build and return the observation space."""
        pass

    @abstractmethod
    def _build_action_space(self) -> spaces.Space:
        """Build and return the action space."""
        pass

    def get_observation(self, states: TensorState | Any) -> np.ndarray | torch.Tensor | dict:
        """Get observation from states using simulator-specific method.

        Args:
            states: Current states from the simulator

        Returns:
            Observation in the format expected by the RL algorithm
        """
        method_name = f"get_observation_{self.sim_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(states)
        else:
            return self._get_observation_generic(states)

    @abstractmethod
    def _get_observation_generic(self, states: TensorState | Any) -> np.ndarray | torch.Tensor | dict:
        """Generic observation extraction (fallback if no simulator-specific method)."""
        pass

    def compute_reward(
        self,
        states: TensorState | Any,
        actions: np.ndarray | torch.Tensor,
        next_states: TensorState | Any,
    ) -> float | np.ndarray | torch.Tensor:
        """Compute reward using simulator-specific method if available.

        Args:
            states: Current states
            actions: Actions taken
            next_states: States after taking actions

        Returns:
            Reward value(s)
        """
        method_name = f"compute_reward_{self.sim_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(states, actions, next_states)
        else:
            return self._compute_reward_generic(states, actions, next_states)

    @abstractmethod
    def _compute_reward_generic(
        self,
        states: TensorState | Any,
        actions: np.ndarray | torch.Tensor,
        next_states: TensorState | Any,
    ) -> float | np.ndarray | torch.Tensor:
        """Generic reward computation."""
        pass

    def check_termination(self, states: TensorState | Any) -> bool | np.ndarray | torch.Tensor:
        """Check if episode should terminate.

        Args:
            states: Current states

        Returns:
            Boolean or array indicating termination
        """
        method_name = f"check_termination_{self.sim_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(states)
        else:
            return self._check_termination_generic(states)

    def _check_termination_generic(self, states: TensorState | Any) -> bool | np.ndarray | torch.Tensor:
        """Generic termination check (default: never terminate)."""
        return False

    def reset_task(self, env_ids: list[int] | None = None):
        """Perform task-specific reset operations.

        Args:
            env_ids: Environment indices to reset (None = reset all)
        """
        method_name = f"reset_task_{self.sim_type}"
        if hasattr(self, method_name):
            getattr(self, method_name)(env_ids)
        else:
            self._reset_task_generic(env_ids)

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset (default: no-op)."""
        return

    def randomize_task(self, env_ids: list[int] | None = None):
        """Perform task randomization for domain randomization.

        Args:
            env_ids: Environment indices to randomize (None = randomize all)
        """
        method_name = f"randomize_task_{self.sim_type}"
        if hasattr(self, method_name):
            getattr(self, method_name)(env_ids)
        else:
            self._randomize_task_generic(env_ids)

    def _randomize_task_generic(self, env_ids: list[int] | None = None):
        """Generic task randomization (default: no-op)."""
        return


class IsaacGymEnvsTaskWrapper(BaseTaskWrapper):
    """Base class for IsaacGymEnvs task wrappers.

    This class provides common functionality for wrapping tasks from
    the IsaacGymEnvs benchmark, handling the specific state representations
    and patterns used by that framework.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        """Initialize IsaacGymEnvs task wrapper.

        Args:
            env: The wrapped environment
            cfg: Task configuration object
            sim_type: Simulator type (default: 'isaacgym')
        """
        super().__init__(env, cfg, sim_type)

        self.num_envs = getattr(env, "num_envs", 1)
        if hasattr(env, "device"):
            self.device = env.device
        elif hasattr(env, "handler") and hasattr(env.handler, "device"):
            self.device = env.handler.device
        else:
            self.device = "cpu"

        self.num_obs = None
        self.num_actions = None
        self.obs_buf = None
        self.reward_buf = None
        self.reset_buf = None
        self.progress_buf = None
        self.randomize_buf = None
        self.extras = {}

    def _get_tensor_state(self, states):
        """Extract TensorState from states object.

        Handles both dictionary states and TensorState objects.
        """
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return states
        elif isinstance(states, dict) and "tensor_state" in states:
            return states["tensor_state"]
        else:
            raise ValueError(f"Cannot extract TensorState from {type(states)}")

    def _create_tensor_if_needed(self, shape, dtype=torch.float, fill_value=0.0):
        """Create a tensor buffer if it doesn't exist."""
        if isinstance(shape, int):
            shape = (shape,)
        tensor = torch.full(shape, fill_value, dtype=dtype, device=self.device)
        return tensor

    def initialize_buffers(self):
        """Initialize common buffers used by IsaacGymEnvs tasks."""
        actual_num_envs = self.num_envs
        if hasattr(self.env, "num_envs"):
            actual_num_envs = self.env.num_envs
        elif hasattr(self.env, "handler") and hasattr(self.env.handler, "num_envs"):
            actual_num_envs = self.env.handler.num_envs

        if self.num_obs is not None and self.obs_buf is None:
            self.obs_buf = self._create_tensor_if_needed((actual_num_envs, self.num_obs))

        if self.reward_buf is None:
            self.reward_buf = self._create_tensor_if_needed(actual_num_envs)

        if self.reset_buf is None:
            self.reset_buf = self._create_tensor_if_needed(actual_num_envs, dtype=torch.bool)

        if self.progress_buf is None:
            self.progress_buf = self._create_tensor_if_needed(actual_num_envs, dtype=torch.long)

        if self.randomize_buf is None:
            self.randomize_buf = self._create_tensor_if_needed(actual_num_envs, dtype=torch.long)
