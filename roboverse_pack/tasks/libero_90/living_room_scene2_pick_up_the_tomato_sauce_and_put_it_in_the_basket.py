"""Configuration for the Libero living room scene2 pick up the tomato sauce and put it in the basket task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
)
class LiberoLivingRoomScene2PickUpTomatoSauceTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene10 put the chocolate pudding in the top drawer of the cabinet and close it task.

    Task Description:
    - Pick up the tomato_sauce from the table
    - Place the tomato_sauce inside the basket/contain_region

    This is a compound manipulation task that requires:
    1. Picking up the tomato_sauce from the table
    2. Placing the tomato_sauce inside the basket/contain_region

    Objects:
    - alphabet_soup
    - cream_cheese
    - tomato_sauce (target)
    - ketchup
    - orange_juice
    - milk
    - butter
    - basket (goal container)

    Goal: Place tomato_sauce inside basket/contain_region.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="tomato_sauce",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/urdf/tomato_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/mjcf/tomato_sauce.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="alphabet_soup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/urdf/alphabet_soup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/mjcf/alphabet_soup.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="cream_cheese",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/urdf/cream_cheese.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/mjcf/cream_cheese.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="ketchup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="orange_juice",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/usd/orange_juice.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/urdf/orange_juice.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/mjcf/orange_juice.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="milk",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/usd/milk.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/urdf/milk.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/mjcf/milk.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="butter",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="basket",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/mjcf/basket.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 200
    task_desc = "Pick up the tomato_sauce and put it in the basket (living_room_scene2)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: tomato_sauce is inside basket/contain_region bounding box."""
        obj_pos = states.objects["tomato_sauce"].root_state[:, :3]
        N = obj_pos.shape[0]
        region_mat = self.handler.physics.named.data.site_xmat["basket/contain_region"]
        region_pos = self.handler.physics.named.data.site_xpos["basket/contain_region"]
        region_R = torch.from_numpy(region_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(obj_pos.device)
        region_t = torch.from_numpy(region_pos).float().unsqueeze(0).expand(N, 3).to(obj_pos.device)
        half_size = torch.tensor([0.06108, 0.06108, 0.06949], device=obj_pos.device)
        obj_local = torch.matmul(region_R.transpose(1, 2), (obj_pos - region_t).unsqueeze(-1)).squeeze(-1)
        inside = (obj_local.abs() <= (half_size + 1e-6)).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
