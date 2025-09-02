from __future__ import annotations

import math

import torch

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState
from metasim.example.example_pack.tasks.checkers.checkers import JointPosChecker


@register_task("rlbench.close_box", "close_box", "franka.close_box")
class CloseBoxTask(BaseTaskEnv):
    episode_length = 250

    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
        robots=["franka"],
    )

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        self.traj_filepath = "roboverse_data/trajs/rlbench/close_box/v2/franka_v2.pkl.gz"
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super()._init_ because handler init
        super().__init__(scenario, device)

        # task horizon
        self.max_episode_steps = 250

        # success checker: cube falls into a bbox above base
        self.checker = JointPosChecker(
            obj_name="box_base",
            joint_name="box_joint",
            mode="le",
            radian_threshold=-14 / 180 * math.pi,
        )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when cube is detected in the bbox above base."""
        return self.checker.check(self.handler, states)

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
