from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class EmbodiedGenBaseTask(BaseTaskEnv):
    """Base class for EmbodiedGen tasks.

    This base class supports both trajectory-based and manually-defined initial states.
    If traj_filepath is None, subclasses should override _get_initial_states() to provide
    manual initial states.
    """

    scenario = None
    max_episode_steps = 250
    checker = None
    traj_filepath = None

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        # Only check and download trajectory if it's specified
        if self.traj_filepath is not None:
            check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super().__init__ because handler init
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the checker conditions are met."""
        if self.checker is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None):
        """Reset the checker."""
        states = super().reset(states, env_ids)
        if self.checker is not None:
            self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states from trajectory file or return None for manual definition.

        If traj_filepath is None, subclasses should override this method to provide
        manual initial states. Otherwise, states are loaded from the trajectory file.
        """
        if self.traj_filepath is None:
            # No trajectory file - return None or let subclass override
            return None

        # Load initial states from trajectory file
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
