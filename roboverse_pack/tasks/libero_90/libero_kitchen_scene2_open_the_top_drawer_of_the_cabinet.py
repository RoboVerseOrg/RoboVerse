"""Configuration for the Libero kitchen scene2 open the top drawer of the cabinet task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import JointPosChecker
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene2_open_the_top_drawer_of_the_cabinet", "kitchen_scene2_open_the_top_drawer_of_the_cabinet"
)
class LiberoKitchenScene2OpenTopDrawerTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene2 open the top drawer of the cabinet task.

    This task is transferred from:
    KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet.bddl

    Task Description:
    - Open the top drawer of the wooden cabinet (scene2)

    This is a manipulation task that requires:
    1. Grasping and manipulating the cabinet drawer (articulated object)
    2. Opening the top drawer to the required position

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1/2/3 (object): Three bowls on the table

    Goal: (Open wooden_cabinet_1_top_region)
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
                name="akita_black_bowl_3",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="plate",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
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

    max_episode_steps = 300
    task_desc = "Open the top drawer of the cabinet (scene2)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    checker = JointPosChecker(
        obj_name="wooden_cabinet",
        joint_name="top_level",
        mode="le",
        radian_threshold=-0.1,
    )

    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene2_open_the_top_drawer_of_the_cabinet_traj_v2.pkl"
    )
