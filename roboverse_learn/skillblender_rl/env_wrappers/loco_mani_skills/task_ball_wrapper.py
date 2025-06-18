"""SkillBlench wrapper for training loco-manipulation skill: FootballShoot"""

# ruff: noqa: F405
from __future__ import annotations

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.types import EnvState
from metasim.utils.humanoid_robot_util import *
from roboverse_learn.skillblender_rl.env_wrappers.base.humanoid_base_wrapper import HumanoidBaseWrapper


class ReachingWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:walking

    # TODO implement push robot.
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        _, _ = self.env.reset()
        self.ori_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _parse_ref_wrist_pos(self, envstate: EnvState):
        envstate.robots[self.robot.name].extra["ref_wrist_pos"] = self.ref_wrist_pos
