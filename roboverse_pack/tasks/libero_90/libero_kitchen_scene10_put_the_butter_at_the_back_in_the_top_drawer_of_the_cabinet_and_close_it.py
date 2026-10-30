"""Configuration for the Libero kitchen scene10 put the butter at the back in the top drawer of the cabinet and close it task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

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
    )

    max_episode_steps = 250
    task_desc = "Put the butter at the back in the top drawer of the cabinet and close it (scene10)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_traj_v2.pkl"

    goal = (
        ("In", "butter_2", "wooden_cabinet", "top_region"),
        ("Close", "wooden_cabinet", "top_level"),
    )
