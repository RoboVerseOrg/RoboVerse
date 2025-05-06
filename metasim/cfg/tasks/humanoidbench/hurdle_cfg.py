"""Hurdle task for humanoid robots.

TODO: Not Implemented because of collision detection issues.
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _HurdleChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class HurdleReward(HumanoidBaseReward):
    """Reward function for the hurdle task."""

    def __init__(self, robot_name="h1"):
        """Initialize the hurdle reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the hurdle reward."""
        return torch.zeros(len(states))


@configclass
class HurdleCfg(HumanoidTaskCfg):
    """Hurdle task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="hurdle",
            mjcf_path="roboverse_data/assets/humanoidbench/hurdle/bracket/mjcf/bracket.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/hurdle/v2/initial_state_v2.json"
    checker = _HurdleChecker()
    reward_weights = [1.0]
    reward_functions = [HurdleReward]
