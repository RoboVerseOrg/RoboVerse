"""Walking up stairs task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    body_pos_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

from .humanoid_env import BaseLocomotionEnv

_WALK_SPEED = 1.0  # m/s, the minimum forward speed for the humanoid to be considered walking


class StairReward:
    """Reward function for the stair task."""

    def __init__(self, robot_name="h1"):
        """Initialize the stair reward."""
        self.robot_name = robot_name

    def __call__(self, states: TensorState) -> torch.FloatTensor:
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


@register_task("humanoid.stair", "stair", "h1.stair")
class StairEnv(BaseLocomotionEnv):
    """Walking up stairs task for humanoid robots."""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="stair",
                mjcf_path="roboverse_data/assets/humanoidbench/stair/floor/mjcf/floor.xml",
                physics=PhysicStateType.GEOM,
                fix_base_link=True,
            ),
        ],
        robots=["h1"],
    )

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        self.max_episode_steps = 1000
        self.reward_functions = [StairReward(self.robot_name)]
        self.reward_weights = [1.0]

    def extra_spec(self):
        """Declare extra observations needed by StairReward."""
        return {
            "head_pos": SitePos("head"),
        }
