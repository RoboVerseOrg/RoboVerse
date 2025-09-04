from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import PositionShiftChecker
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


@register_task("maniskill.pick_cube", "pick_cube")
class PickCubeTask(BaseTaskEnv):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            )
        ],
        robots=["franka"],
    )

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        self.traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super().__init__ because handler init
        super().__init__(scenario, device)

        # task horizon
        self.max_episode_steps = 250

        # success checker: cube is lifted by at least 0.1m along z-axis
        self.checker = PositionShiftChecker(
            obj_name="cube",
            distance=0.1,
            axis="z",
        )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the cube has been lifted sufficiently along z from its reset height."""
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None):
        """Reset the checker."""
        states = super().reset(states, env_ids)
        self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Give the inital states from traj file."""
        # Keep it simple and leave robot states to defaults; just seed cube pose.
        # If the handler handles None gracefully, this can be set to None.
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
