"""ManiSkill Pick-Cube task as a runnable BaseTaskEnv.

This refactors the previous config-only version to a task class that:
- Inherits the common BaseTaskEnv used by get_started examples
- Uses `_load_task_config` to populate scenario objects
- Implements `_terminated` to replace the old checker-based success logic

Success condition: the `cube` is lifted by at least 0.1 m along the z-axis from its reset height.
"""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import FileDownloader
from metasim.utils.state import TensorState
from roboverse_pack.tasks.maniskill.checkers.checkers import PositionShiftChecker


@register_task("maniskill.pick_cube", "pick_cube", "franka.pick_cube")
class PickCubeTask(BaseTaskEnv):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        self.traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2"
        FileDownloader(self).do_it()
        super().__init__(scenario, device)

    def _load_task_config(self, scenario: ScenarioCfg) -> ScenarioCfg:
        """Populate objects and episode length similar to get_started exaples."""
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
            )
        ]
        self.checker = PositionShiftChecker(
            obj_name="cube",
            distance=0.1,
            axis="z",
        )

        return scenario

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
        return initial_states

    # def _observation(self, states: TensorState) -> torch.Tensor:
    #     """Return a simple observation: robot joint positions and end-effector position if available."""
    #     if not states.robots:
    #         return torch.empty((self.handler.num_envs, 0), device=self.handler.device)
    #     # use first robot
    #     robot_name, robot_state = next(iter(sorted(states.robots.items())))
    #     joint_pos = (
    #         robot_state.joint_pos
    #         if robot_state.joint_pos is not None
    #         else torch.empty((self.handler.num_envs, 0), device=self.handler.device)
    #     )
    #     ee_pos = None
    #     if robot_state.body_state is not None and robot_state.body_names:
    #         try:
    #             hand_idx = robot_state.body_names.index("panda_hand")
    #         except ValueError:
    #             hand_idx = 0
    #         ee_pos = robot_state.body_state[:, hand_idx, :3]
    #     else:
    #         ee_pos = torch.empty((self.handler.num_envs, 0), device=self.handler.device)
    #     return (
    #         torch.cat([joint_pos, ee_pos], dim=1)
    #         if joint_pos.numel() or ee_pos.numel()
    #         else torch.empty((self.handler.num_envs, 0), device=self.handler.device)
    #     )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the cube has been lifted sufficiently along z from its reset height."""
        self.checker.check(self.handler, states)

    def reset(self, env_ids=None):
        """Reset the checker."""
        states = super().reset(env_ids)
        self.checker.reset(self.handler)
        return states

    def _time_out(self, env_states) -> torch.Tensor:
        """Timeout flags."""
        return (self.max_episode_steps >= self.max_episode_steps).to(dtype=torch.bool, device=self.device)
