"""Configuration for the Libero kitchen open bottom drawer task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import JointPosChecker
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task("libero_90.kitchen_scene1_open_bottom_drawer", "kitchen_open_bottom_drawer")
class LiberoKitchenOpenBottomDrawerTask(Libero90BaseTask):
    """Configuration for the Libero kitchen open bottom drawer task.

    This task is transferred from:
    KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl

    Task Description:
    - Open the bottom drawer of the wooden cabinet

    This is a manipulation task that requires:
    1. Grasping and manipulating the cabinet drawer (articulated object)
    2. Opening the bottom drawer to the required positionw2

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1 (object): Bowl on the table
    - plate_1 (object): Plate on the table

    Goal: (Open wooden_cabinet_1_bottom_region)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects (from BDDL :objects section)
            RigidObjCfg(
                name="akita_black_bowl",
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
            # Fixed fixtures (from BDDL :fixtures section)
            # Note: kitchen_table is handled by the scene/arena
            ArticulationObjCfg(
                name="wooden_cabinet",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/usd/wooden_cabinet.usda",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/urdf/wooden_cabinet.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/mjcf/wooden_cabinet.xml",
            ),
        ],
        robots=["franka"],
        # Scene configuration (from BDDL problem domain)
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    # Task parameters
    max_episode_steps = 300  # Moderate complexity task
    task_desc = "Open the bottom drawer of the cabinet"

    # Workspace configuration (from BDDL regions)
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Checker: bottom drawer must be open
    # Use JointPosChecker to check if the drawer joint position exceeds threshold
    checker = JointPosChecker(
        obj_name="wooden_cabinet",
        joint_name="bottom_level",  # Assuming this is the joint name
        mode="le",  # greater than or equal to
        radian_threshold=-0.1,  # Minimum opening distance (in radians)
    )

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene1_traj_v2.pkl"

    # # rewrite terminate
    # def _terminated(self, states: TensorState) -> torch.Tensor:
    #     """No terminate condition yet. Will terminate when time is up."""
    #     return torch.tensor([False])

    # # rewrite checker
    # def reset(self, states=None, env_ids=None):
    #     """Skip checker reset."""
    #     states = super(Libero90BaseTask, self).reset(states, env_ids)
    #     return states
