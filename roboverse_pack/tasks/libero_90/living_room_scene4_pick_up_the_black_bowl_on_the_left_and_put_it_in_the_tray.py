"""Configuration for the Libero living room scene4 pick up the black bowl on the left and put it in the tray task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
    "living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
)
class LiberoLivingRoomScene4PickUpLeftBlackBowlTask(Libero90BaseTask):
    """Configuration for the Libero living room scene4 pick up the black bowl on the left and put it in the tray task.

    Task Description:
    - Pick up the akita_black_bowl_1 (left bowl) from the table
    - Place the akita_black_bowl_1 inside the wooden_tray/contain_region

    This is a manipulation task that requires:
    1. Picking up the akita_black_bowl_1 from the table
    2. Placing the akita_black_bowl_1 inside the wooden_tray/contain_region

    Objects:
    - akita_black_bowl_1 (target, left bowl)
    - akita_black_bowl_2 (right bowl)
    - chocolate_pudding
    - new_salad_dressing
    - wooden_tray (goal container)

    Goal: Place akita_black_bowl_1 inside wooden_tray/contain_region.
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

    max_episode_steps = 200
    task_desc = "Pick up the black bowl on the left and put it in the tray (living_room_scene4)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: akita_black_bowl_1 is inside wooden_tray/contain_region bounding box.

        Site wooden_tray/contain_region half-size = 0.04038 0.07839 0.13549.
        We transform the akita_black_bowl_1 position into the site's local frame and verify it falls within +/- half-size.
        """
        obj_pos = states.objects["akita_black_bowl_1"].root_state[:, :3]
        N = obj_pos.shape[0]
        region_mat = self.handler.physics.named.data.site_xmat["wooden_tray/contain_region"]
        region_pos = self.handler.physics.named.data.site_xpos["wooden_tray/contain_region"]
        region_R = torch.from_numpy(region_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(obj_pos.device)
        region_t = torch.from_numpy(region_pos).float().unsqueeze(0).expand(N, 3).to(obj_pos.device)
        half_size = torch.tensor([0.04038, 0.07839, 0.13549], device=obj_pos.device)
        obj_local = torch.matmul(region_R.transpose(1, 2), (obj_pos - region_t).unsqueeze(-1)).squeeze(-1)
        inside = (obj_local.abs() <= (half_size + 1e-6)).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
