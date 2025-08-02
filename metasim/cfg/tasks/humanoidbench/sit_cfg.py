"""Sit task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _SitChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    robot_position_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class SitReward(HumanoidBaseReward):
    """Vectorised reward for the sit task (batch-first TensorState)."""

    def __init__(self, robot_name: str = "h1"):
        super().__init__(robot_name)

    def __call__(self, states: dict) -> torch.FloatTensor:
        """Vectorised HumanoidBench-style sitting reward.

        Returns:
        -------
        reward : (B,) tensor
        info   : dict   # individual terms for logging
        """
        # ------------------------------------------------------------------
        # Core state tensors
        # ------------------------------------------------------------------
        robot_pos = robot_position_tensor(states, self.robot_name)  # (B, 3)
        chair_pos = states.objects["sit"].root_state[:, 0:3]  # (B, 3)

        head_z = states.extras["head_pos"][:, 2]
        imu_z = states.extras["imu_pos"][:, 2]  # (B,)

        vel_xy = robot_velocity_tensor(states, self.robot_name)[:, :2]  # (B, 2)
        vx, vy = vel_xy[:, 0], vel_xy[:, 1]

        # ------------------------------------------------------------------
        # 1) Sitting on chair
        # ------------------------------------------------------------------
        sitting = tolerance_tensor(  # height within [0.68, 0.72]
            robot_pos[:, 2], bounds=(0.68, 0.72), margin=0.2
        )

        on_chair_x = tolerance_tensor(  # X aligned with chair
            robot_pos[:, 0] - chair_pos[:, 0], bounds=(-0.19, 0.19), margin=0.2
        )
        on_chair_y = tolerance_tensor(  # Y aligned with chair
            robot_pos[:, 1] - chair_pos[:, 1], margin=0.1
        )
        on_chair = on_chair_x * on_chair_y

        # ------------------------------------------------------------------
        # 2) Posture & upright
        # ------------------------------------------------------------------
        sitting_posture = tolerance_tensor(head_z - imu_z, bounds=(0.35, 0.45), margin=0.3)

        upright = tolerance_tensor(
            torso_upright_tensor(states, self.robot_name),
            bounds=(0.95, float("inf")),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0.0,
        )

        sit_reward = 0.5 * sitting + 0.5 * on_chair  # blend
        sit_reward = sit_reward * upright * sitting_posture  # pose-conditioned

        # ------------------------------------------------------------------
        # 3) Control effort penalty → favour “small_control”
        # ------------------------------------------------------------------
        small_control = tolerance_tensor(
            actuator_forces_tensor(states, self.robot_name),  # (B, n_act)
            margin=10.0,
            value_at_margin=0.0,
            sigmoid="quadratic",
        ).mean(dim=-1)  # → (B,)
        small_control = (4.0 + small_control) / 5.0  # rescale ~[0.8,1]

        # ------------------------------------------------------------------
        # 4) Don’t move horizontally
        # ------------------------------------------------------------------
        dont_move = tolerance_tensor(
            torch.stack([vx, vy], dim=-1),  # (B, 2)
            margin=2.0,
        ).mean(dim=-1)  # → (B,)

        # ------------------------------------------------------------------
        # 5) Final reward & info dict
        # ------------------------------------------------------------------
        reward = small_control * sit_reward * dont_move  # (B,)
        return reward


@configclass
class SitCfg(HumanoidTaskCfg):
    """Sit task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="sit",
            mjcf_path="roboverse_data/assets/humanoidbench/sit/chair/mjcf/chair.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/sit/v2/initial_state_v2.json"
    checker = _SitChecker()
    reward_weights = [1.0]
    reward_functions = [SitReward]

    def extra_spec(self):
        """Declare extra observations needed by CrawlReward."""
        return {
            "imu_pos": SitePos("imu"),
            "head_pos": SitePos("head"),
        }
