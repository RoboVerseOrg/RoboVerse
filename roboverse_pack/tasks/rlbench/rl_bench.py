from __future__ import annotations

import torch

from metasim.example.example_pack.tasks.checkers import EmptyChecker
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class RLBenchTask(BaseTaskEnv):
    max_episode_steps = 250

    scenario = ScenarioCfg(
        objects=[],
        robots=["franka"],
    )

    traj_filepath = None

    checker = EmptyChecker()

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super()._init_ because handler init
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when cube is detected in the bbox above base."""
        return self.checker.check(self.handler, states)

    def _get_initial_states(self) -> list[dict] | None:
        """Give the inital states from traj file."""
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states

    def reset(self, states=None, env_ids=None):
        """Reset the checker."""
        states = super().reset(states, env_ids)
        self.checker.reset(self.handler, env_ids=env_ids)
        return states
