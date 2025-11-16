"""Configuration for the Libero kitchen open drawer put bowl task."""

from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class Libero90BaseTask(BaseTaskEnv):
    """Base class for LIBERO 90 tasks.

    This base class handles the common functionality for LIBERO 90 tasks,
    which are more complex than LIBERO Object tasks.
    """

    scenario = None
    max_episode_steps = 300  # LIBERO 90 tasks are more complex
    task_desc = None
    checker = None
    traj_filepath = None
    decimation = 30

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super()._init_ because handler init
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when task conditions are met."""
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None):
        """Reset the checker."""
        states = super().reset(states, env_ids)
        self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Give the initial states from traj file."""
        # Keep it simple and leave robot states to defaults; just seed object poses.
        # If the handler handles None gracefully, this can be set to None.
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
