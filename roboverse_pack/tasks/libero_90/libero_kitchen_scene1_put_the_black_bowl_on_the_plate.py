"""Configuration for the Libero kitchen put the black bowl on the plate task."""

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
    "libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate", "kitchen_scene1_put_the_black_bowl_on_the_plate"
)
class LiberoKitchen1PutBowlOnPlateTask(Libero90BaseTask):
    """Configuration for the Libero kitchen put the black bowl on the plate task.

    This task is transferred from:
    KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl

    Task Description:
    - Put the akita black bowl on the plate

    This is a manipulation task that requires:
    1. Grasping the akita black bowl
    2. Placing it on top of the plate

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1 (object): Bowl to be placed on the plate
    - plate_1 (object): Target plate

    Goal: (On akita_black_bowl_1 plate_1)
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
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/usd/wooden_cabinet.usd",
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
    task_desc = "Put the black bowl on the plate"

    # Workspace configuration (from BDDL regions)
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Checker: bowl must be on the plate (within detection region above plate)
    # checker = DetectedChecker(
    #     obj_name="akita_black_bowl",
    #     detector=RelativeBboxDetector(
    #         base_obj_name="plate",
    #         relative_pos=(0.0, 0.0, 0.05),  # Slightly above the plate center
    #         relative_quat=(1.0, 0.0, 0.0, 0.0),  # Identity rotation
    #         checker_lower=(-0.1, -0.1, -0.02),  # Detection region half-size
    #         checker_upper=(0.1, 0.1, 0.1),
    #         ignore_base_ori=True,
    #         debug_vis=False,
    #         name="plate_region",
    #         fixed=True,
    #     ),
    # )

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene1_put_the_black_bowl_on_the_plate_traj_v2.pkl"

    # env.handler.get_states[0]['objects']['wooden_cabinet']['dof_pos']["top_level"] 获取top
    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        bowl_pos = states.objects["akita_black_bowl"].root_state[:, :3]  # (N,3)
        plate_pos = states.objects["plate"].root_state[:, :3]  # (N,3)
        # Check if bowl is within a small region above the plate
        range_threshold = 0.06  # Radius of the range in xy plane
        height_threshold = 0.03  # Height threshold above the plate

        # Calculate xy distance between bowl and plate
        xy_distance = torch.norm(bowl_pos[:, :2] - plate_pos[:, :2], dim=-1)  # (N,)
        # Calculate height difference (bowl z - plate z)
        height_diff = bowl_pos[:, 2] - plate_pos[:, 2]  # (N,)
        # Check both conditions: xy distance < range AND 0 < height_diff < height_threshold
        xy_close = xy_distance < range_threshold  # (N,)
        height_valid = (height_diff > 0) & (height_diff < height_threshold)  # (N,)

        is_on_plate = xy_close & height_valid  # (N,)
        return is_on_plate

    # rewrite checker
    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
