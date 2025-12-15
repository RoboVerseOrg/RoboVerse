"""Configuration for the Libero living room scene4 stack the left bowl on the right bowl and place them in the tray task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    "living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
)
class LiberoLivingRoomScene4StackLeftBowlOnRightBowlTask(Libero90BaseTask):
    """Configuration for the Libero living room scene4 stack the left bowl on the right bowl and place them in the tray task.

    Task Description:
    - Stack akita_black_bowl_1 (left bowl) on top of akita_black_bowl_2 (right bowl)
    - Place the stacked bowls inside the wooden_tray/contain_region

    This is a compound manipulation task that requires:
    1. Stacking akita_black_bowl_1 on top of akita_black_bowl_2
    2. Placing both bowls inside the wooden_tray/contain_region

    Objects:
    - akita_black_bowl_1 (left bowl, to be stacked on top)
    - akita_black_bowl_2 (right bowl, base)
    - chocolate_pudding
    - new_salad_dressing
    - wooden_tray (goal container)

    Goal: Both akita_black_bowl_1 and akita_black_bowl_2 inside wooden_tray/contain_region, with bowl_1 stacked on bowl_2.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="akita_black_bowl_1",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl_2",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chocolate_pudding",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/usd/chocolate_pudding.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/urdf/chocolate_pudding.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/mjcf/chocolate_pudding.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="new_salad_dressing",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/usd/new_salad_dressing.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/urdf/new_salad_dressing.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/mjcf/new_salad_dressing.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wooden_tray",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/usd/wooden_tray.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/urdf/wooden_tray.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/mjcf/wooden_tray.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 350
    task_desc = "Stack the left bowl on the right bowl and place them in the tray (living_room_scene4)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: bowl_1 is stacked on bowl_2 and bowl_2 is inside wooden_tray/contain_region.

        Site wooden_tray/contain_region half-size = 0.04038 0.07839 0.13549.
        We verify that:
        1. akita_black_bowl_1 is stacked on akita_black_bowl_2 (xy distance < 0.06m and 0 < height_diff < 0.05m)
        2. akita_black_bowl_2 is inside wooden_tray/contain_region
        """
        bowl1_pos = states.objects["akita_black_bowl_1"].root_state[:, :3]  # (N,3)
        bowl2_pos = states.objects["akita_black_bowl_2"].root_state[:, :3]  # (N,3)
        N = bowl1_pos.shape[0]

        region_mat = self.handler.physics.named.data.site_xmat["wooden_tray/contain_region"]
        region_pos = self.handler.physics.named.data.site_xpos["wooden_tray/contain_region"]
        region_R = torch.from_numpy(region_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(bowl1_pos.device)
        region_t = torch.from_numpy(region_pos).float().unsqueeze(0).expand(N, 3).to(bowl1_pos.device)
        half_size = torch.tensor([0.04038, 0.07839, 0.13549], device=bowl1_pos.device)

        # Check if bowl2 is inside tray
        bowl2_local = torch.matmul(region_R.transpose(1, 2), (bowl2_pos - region_t).unsqueeze(-1)).squeeze(-1)
        bowl2_inside = (bowl2_local.abs() <= (half_size + 1e-6)).all(dim=-1)

        # Check if bowl1 is stacked on bowl2 with xy distance and height thresholds
        range_threshold = 0.06  # Radius of the range in xy plane
        height_threshold = 0.05  # Height threshold above bowl_2
        xy_distance = torch.norm(bowl1_pos[:, :2] - bowl2_pos[:, :2], dim=-1)  # (N,)
        height_diff = bowl1_pos[:, 2] - bowl2_pos[:, 2]  # (N,)
        xy_close = xy_distance < range_threshold  # (N,)
        height_valid = (height_diff > 0) & (height_diff < height_threshold)  # (N,)
        is_stacked = xy_close & height_valid  # (N,)

        return bowl2_inside & is_stacked

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
