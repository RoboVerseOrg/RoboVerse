"""Crawl task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _CrawlChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    robot_rotation_tensor,
    robot_velocity_tensor,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class CrawlReward(HumanoidBaseReward):
    """Implements the HumanoidBench Crawl reward with batched TensorState."""

    def __init__(self, robot_name="h1"):
        """Initialize the crawl reward."""
        super().__init__(robot_name)

        # Define the expected quaternion for crawling, normalized
        quat_raw = torch.tensor([0.75, 0, 0.65, 0])
        self.quat_crawl = quat_raw / torch.norm(quat_raw)

    def __call__(self, states):
        """Vectorised HumanoidBench-style sitting reward."""
        # ---------------- core state tensors -------------------------     # (B, 3)
        imu_pos = states.extras["imu_pos"]  # (B, 3)

        pelvis_q = robot_rotation_tensor(states, self.robot_name)  # (B, 4)
        com_vx = robot_velocity_tensor(states, self.robot_name)[:, 0]  # (B,)

        # ---------------- 1) small-control term -----------------------
        small_ctrl = tolerance_tensor(
            actuator_forces_tensor(states, self.robot_name),  # (B, n_act)
            margin=10.0,
            value_at_margin=0.0,
            sigmoid="quadratic",
        ).mean(dim=-1)  # (B,)
        small_ctrl = (4.0 + small_ctrl) / 5.0  # (B,)

        # ---------------- 2) forward-motion term ----------------------
        move = tolerance_tensor(
            com_vx,
            bounds=(1.0, float("inf")),
            margin=1.0,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        move = (5.0 * move + 1.0) / 6.0  # (B,)

        # ---------------- 3) crawling-height terms --------------------
        height_bounds = (self._crawl_height - 0.2, self._crawl_height + 0.2)

        head_z = states.extras["head_pos"][:, 2]
        crawling_head = tolerance_tensor(head_z, bounds=height_bounds, margin=1.0)

        crawling_imu = tolerance_tensor(imu_pos[:, 2], bounds=height_bounds, margin=1.0)

        # --- one-time device alignment for quat_crawl --------------
        if self.quat_crawl.device != pelvis_q.device:  # only first mismatch
            self.quat_crawl = self.quat_crawl.to(pelvis_q.device)

        # ---------------- 4) pelvis orientation term ------------------
        quat_diff = torch.norm(pelvis_q - self.quat_crawl[None, :], dim=-1)
        reward_xquat = tolerance_tensor(quat_diff, margin=1.0)  # bounds default (0,0)

        # ---------------- 5) tunnel constraint ------------------------
        in_tunnel = tolerance_tensor(imu_pos[:, 1], bounds=(-1.0, 1.0), margin=0.0)

        # ---------------- 6) final reward -----------------------------
        min_height = torch.minimum(crawling_head, crawling_imu)  # element-wise

        reward = (0.1 * small_ctrl + 0.25 * min_height + 0.4 * move + 0.25 * reward_xquat) * in_tunnel  # (B,)

        return reward


@configclass
class CrawlCfg(HumanoidTaskCfg):
    """Crawl task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="tunnel",
            mjcf_path="roboverse_data/assets/humanoidbench/crawl/tunnel/mjcf/tunnel.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/crawl/v2/initial_state_v2.json"
    checker = _CrawlChecker()
    reward_weights = [1.0]
    reward_functions = [CrawlReward]

    def extra_spec(self):
        """Declare extra observations needed by CrawlReward."""
        return {
            "imu_pos": SitePos("imu"),
            "head_pos": SitePos("head"),
        }
