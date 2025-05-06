"""Going through poles task for humanoid robots.

TODO: Not Implemented because of collision detection issues.
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _PoleChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class PoleReward(HumanoidBaseReward):
    """Reward function for the pole task."""

    def __init__(self, robot_name="h1"):
        """Initialize the pole reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the pole reward."""
        return torch.zeros(len(states))


@configclass
class PoleCfg(HumanoidTaskCfg):
    """Going through poles task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="floor",
            mjcf_path="roboverse_data/assets/humanoidbench/pole/floor/mjcf/floor.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/pole/v2/initial_state_v2.json"
    checker = _PoleChecker()
    reward_weights = [1.0]
    reward_functions = [PoleReward]
