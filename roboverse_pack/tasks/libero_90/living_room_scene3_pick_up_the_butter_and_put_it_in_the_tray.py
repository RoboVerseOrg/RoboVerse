"""Configuration for the Libero living room scene3 pick up the butter and put it in the tray task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray",
    "living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray",
)
class LiberoLivingRoomScene3PickUpButterTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene10 put the chocolate pudding in the top drawer of the cabinet and close it task.

    Task Description:
    - Pick up the butter from the table
    - Place the butter inside the tray/contain_region

    This is a compound manipulation task that requires:
    1. Picking up the butter from the table
    2. Placing the butter inside the tray/contain_region

    Objects:
    - alphabet_soup
    - cream_cheese
    - tomato_sauce
    - ketchup
    - butter (target)
    - wooden_tray (goal container)

    Goal: Place butter inside wooden_tray.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="butter",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
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
                name="tomato_sauce",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/urdf/tomato_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/mjcf/tomato_sauce.xml",
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
                name="wooden_tray",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/usd/wooden_tray.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/urdf/wooden_tray.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/mjcf/wooden_tray.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 131
    task_desc = "Pick up the butter and put it in the tray (living_room_scene3)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0.42),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: butter is inside wooden_tray/contain_region bounding box."""
        obj_pos = states.objects["butter"].root_state[:, :3]
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
