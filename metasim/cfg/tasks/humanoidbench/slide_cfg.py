"""Walking on slide task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _SlideChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.types import EnvState
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    body_pos_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

_WALK_SPEED = 1.0

from .base_cfg import (
    HumanoidBaseReward,
    HumanoidTaskCfg,
)


class SlideReward(HumanoidBaseReward):
    """Reward function for the slide task."""

    def __init__(self, robot_name="h1"):
        """Initialize the slide reward."""
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState]) -> torch.FloatTensor:
        """Compute the stair reward."""
        forces = actuator_forces_tensor(states, self.robot_name)  # (B, n_act)
        com_vx = robot_velocity_tensor(states, self.robot_name)[:, 0]  # (B,)
        upright_ = torso_upright_tensor(states, self.robot_name)  # (B,)

        head_z = states.extras["head_pos"][:, 2]
        lfoot_z = body_pos_tensor(states, self.robot_name, "left_ankle_link")[:, 2]  # (B,)
        rfoot_z = body_pos_tensor(states, self.robot_name, "right_ankle_link")[:, 2]  # (B,)

        # standing term -------------------------------------------------
        standing = tolerance_tensor(head_z - lfoot_z, bounds=(1.2, float("inf")), margin=0.45) * tolerance_tensor(
            head_z - rfoot_z, bounds=(1.2, float("inf")), margin=0.45
        )  # (B,)

        # upright term --------------------------------------------------
        upright = tolerance_tensor(
            upright_,
            bounds=(0.5, float("inf")),
            margin=1.9,
            sigmoid="linear",
            value_at_margin=0.0,
        )  # (B,)

        stand_reward = standing * upright  # (B,)

        # small-control term -------------------------------------------
        small_ctrl = tolerance_tensor(
            forces,
            margin=10.0,
            value_at_margin=0.0,
            sigmoid="quadratic",
        ).mean(dim=-1)  # (B,)
        small_ctrl = (4.0 + small_ctrl) / 5.0  # (B,)

        # forward motion term -------------------------------------------
        move = tolerance_tensor(
            com_vx,
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        move = (5.0 * move + 1.0) / 6.0  # (B,)

        # final reward ---------------------------------------------------
        reward = stand_reward * small_ctrl * move  # (B,)
        return reward


@configclass
class SlideCfg(HumanoidTaskCfg):
    """Slide task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="slide",
            mjcf_path="roboverse_data/assets/humanoidbench/slide/floor/mjcf/floor.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/slide/v2/initial_state_v2.json"
    checker = _SlideChecker()
    reward_weights = [1.0]
    reward_functions = [SlideReward]

    def extra_spec(self):
        """Declare extra observations needed by CrawlReward."""
        return {
            "head_pos": SitePos("head"),
        }
