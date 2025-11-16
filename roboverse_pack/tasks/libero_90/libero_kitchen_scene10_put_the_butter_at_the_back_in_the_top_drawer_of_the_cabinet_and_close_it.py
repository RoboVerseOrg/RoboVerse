"""Configuration for the Libero kitchen scene10 put the butter at the back in the top drawer of the cabinet and close it task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    "kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
)
class LiberoKitchenScene10PutButterInTopDrawerAndCloseTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene10 put the butter at the back in the top drawer of the cabinet and close it task.

    Task Description:
    - Put butter_2 at the back into the top drawer of the wooden cabinet
    - Close the top drawer of the wooden cabinet

    This is a compound manipulation task that requires:
    1. Picking up the butter_2 from the table
    2. Placing the butter_2 inside the top drawer
    3. Closing the top drawer

    Objects:
    - akita_black_bowl (object): Distractor object
    - chocolate_pudding (object): Distractor object
    - butter_1 (object): Distractor object
    - butter_2 (object): Butter to be placed in the drawer
    - wooden_cabinet (fixture): Cabinet with drawers

    Goal: (In butter_2 wooden_cabinet_1_top_region) AND (Close wooden_cabinet_1_top_region)
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="akita_black_bowl",
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
                name="butter_1",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="butter_2",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="wooden_cabinet",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/usd/wooden_cabinet.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/urdf/wooden_cabinet.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/mjcf/wooden_cabinet.xml",
            ),
        ],
        robots=["franka"],
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    max_episode_steps = 250
    task_desc = "Put the butter at the back in the top drawer of the cabinet and close it (scene10)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: butter_2 is inside the top drawer AND top drawer is closed."""
        # 1. check if butter_2 is in the top drawer
        butter_pos = states.objects["butter_2"].root_state[:, :3]  # (N,3)
        N = butter_pos.shape[0]
        # Get drawer top_region site pose and expand to N environments
        drawer_top_mat = self.handler.physics.named.data.site_xmat["wooden_cabinet/top_region"]  # (9,)
        drawer_top_pos = self.handler.physics.named.data.site_xpos["wooden_cabinet/top_region"]  # (3,)

        # Expand to (N,3,3) and (N,3)
        drawer_top_R = (
            torch.from_numpy(drawer_top_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(butter_pos.device)
        )  # (N,3,3)
        drawer_top_t = torch.from_numpy(drawer_top_pos).float().unsqueeze(0).expand(N, 3).to(butter_pos.device)  # (N,3)

        # top_region site half-size from wooden_cabinet.xml: size="0.02993 0.07561 0.10224"
        bbox_lower = torch.tensor([-0.02993, -0.07561, -0.10224], device=butter_pos.device)
        bbox_upper = torch.tensor([0.02993, 0.07561, 0.10224], device=butter_pos.device)

        # Transform butter position to drawer top_region local frame
        butter_local = torch.matmul(drawer_top_R.transpose(1, 2), (butter_pos - drawer_top_t).unsqueeze(-1)).squeeze(
            -1
        )  # (N,3)
        ge_lower = butter_local >= bbox_lower  # (N,3)
        le_upper = butter_local <= bbox_upper  # (N,3)
        butter_in_drawer = (ge_lower & le_upper).all(dim=-1)  # (N,)

        # 2. check if the top drawer is closed
        cabinet_joint_pos = states.objects["wooden_cabinet"].joint_pos[:, 2]  # (N,) - top_level joint (index 2)
        drawer_closed = cabinet_joint_pos > -0.01  # (N,) - closed when joint_pos is near 0

        # Final success condition
        is_success = butter_in_drawer & drawer_closed  # (N,)
        return is_success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
