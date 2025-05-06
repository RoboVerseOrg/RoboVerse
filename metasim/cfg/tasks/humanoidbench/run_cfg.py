"""Run task for humanoid robots."""

from metasim.cfg.checkers import _RunChecker
from metasim.utils import configclass

from .base_cfg import BaseLocomotionReward, HumanoidTaskCfg


class RunReward(BaseLocomotionReward):
    """Reward function for the run task."""

    _move_speed = 5
    success_bar = 700


@configclass
class RunCfg(HumanoidTaskCfg):
    """Run task for humanoid robots."""

    episode_length = 1000
    traj_filepath = "roboverse_data/trajs/humanoidbench/run/v2/initial_state_v2.json"
    checker = _RunChecker()
    reward_functions = [RunReward]
    reward_weights = [1.0]
