from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState
from roboverse_pack.tasks.maniskill.checkers.checkers import DetectedChecker
from roboverse_pack.tasks.maniskill.checkers.detectors import RelativeBboxDetector


@register_task("maniskill.stack_cube", "stack_cube")
class StackCubeTask(BaseTaskEnv):
    """Stack a red cube on top of a blue base cube and release it."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        self.traj_filepath = "roboverse_data/trajs/maniskill/stack_cube/v2"
        super().__init__(scenario, device)

    def _load_task_config(self, scenario: ScenarioCfg) -> ScenarioCfg:
        scenario = super()._load_task_config(scenario)

        # task horizon
        self.max_episode_steps = 250

        scenario.objects = [
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCubeCfg(
                name="base",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.0, 0.0, 1.0),
            ),
        ]

        # success checker: cube falls into a bbox above base
        self.checker = DetectedChecker(
            obj_name="cube",
            detector=RelativeBboxDetector(
                base_obj_name="base",
                relative_pos=(0.0, 0.0, 0.04),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.02, -0.02, -0.02),
                checker_upper=(0.02, 0.02, 0.02),
                ignore_base_ori=True,
            ),
        )
        self.scenario = scenario
        return scenario

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when cube is detected in the bbox above base."""
        return self.checker.check(self.env, states)

    def reset(self, env_ids):
        """Reset the checker."""
        states = super().reset(env_ids)
        self.checker.reset(self.env, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Give the inital states from traj file."""
        # Keep it simple and leave robot states to defaults; just seed cube pose.
        # If the handler handles None gracefully, this can be set to None.
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.env)
        # Duplicate / trim list so that its length matches num_envs
        initial_states = initial_states * self.num_envs
        return initial_states
