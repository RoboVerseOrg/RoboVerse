"""Push cube on table task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _PushChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import left_hand_position

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class SuccessReward(HumanoidBaseReward):
    """Reward function for the push task when the object is pushed to the destination."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the push reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> float:
        """Compute the push reward."""
        state = states[0]
        box_pos = state["metasim_body_object/object"]["pos"]
        dest_pos = state["metasim_site_destination/destination"]["pos"]
        dgoal = torch.norm(box_pos - dest_pos)
        reward_success = float(dgoal < 0.01)
        return reward_success


class GoalDistanceReward(HumanoidBaseReward):
    """Reward function for the push task when the object is getting close to the destination."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the push reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the push reward."""
        state = states[0]
        box_pos = state["metasim_body_object/object"]["pos"]
        dest_pos = state["metasim_site_destination/destination"]["pos"]
        reward_goal = -torch.norm(box_pos - dest_pos)
        return reward_goal


class HandDistanceReward(HumanoidBaseReward):
    """Reward function for the push task when the hand is getting close to the object."""

    def __init__(self, robot_name="h1_simple_hand"):
        """Initialize the push reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the push reward."""
        state = states[0]
        box_pos = state["metasim_body_object/object"]["pos"]
        left_hand_pos = left_hand_position(state)
        reward_hand = -torch.norm(box_pos - left_hand_pos)
        return reward_hand


@configclass
class PushCfg(HumanoidTaskCfg):
    """Push cube on table task for humanoid robots."""

    episode_length = 500
    objects = [
        RigidObjCfg(
            name="table",
            mjcf_path="roboverse_data/assets/humanoidbench/push/table/mjcf/table.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="object",
            mjcf_path="roboverse_data/assets/humanoidbench/push/object/mjcf/object.xml",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="destination",
            mjcf_path="roboverse_data/assets/humanoidbench/push/destination/mjcf/destination.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/push/v2/initial_state_v2.json"
    checker = _PushChecker()
    reward_weights = [1000, 1, 0.1]  # αs, αt, αh
    reward_functions = [SuccessReward, GoalDistanceReward, HandDistanceReward]
