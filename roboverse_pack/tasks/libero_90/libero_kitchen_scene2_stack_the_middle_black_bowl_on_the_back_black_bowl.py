"""Configuration for the Libero kitchen scene2 stack the middle black bowl on the back black bowl task."""

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
    "libero_90.kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl",
)
class LiberoKitchenScene2StackMiddleBowlOnBackBowlTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene2 stack the middle black bowl on the back black bowl task.

    This task is transferred from:
    KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl

    Task Description:
    - Stack akita_black_bowl_2 (middle) on top of akita_black_bowl_3 (back)

    This is a manipulation task that requires:
    1. Grasping akita_black_bowl_2
    2. Placing it on top of akita_black_bowl_3

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1/2/3 (object): Three bowls on the table
    - plate_1 (object): Distractor plate

    Goal: (On akita_black_bowl_2 akita_black_bowl_3)
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
    task_desc = "Stack the middle black bowl (bowl_2) on the back black bowl (bowl_3) (scene2)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        bowl2_pos = states.objects["akita_black_bowl_2"].root_state[:, :3]  # (N,3)
        bowl3_pos = states.objects["akita_black_bowl_3"].root_state[:, :3]  # (N,3)
        range_threshold = 0.06  # Radius of the range in xy plane
        height_threshold = 0.03  # Height threshold above bowl_3
        xy_distance = torch.norm(bowl2_pos[:, :2] - bowl3_pos[:, :2], dim=-1)  # (N,)
        height_diff = bowl2_pos[:, 2] - bowl3_pos[:, 2]  # (N,)
        xy_close = xy_distance < range_threshold  # (N,)
        height_valid = (height_diff > 0) & (height_diff < height_threshold)  # (N,)
        is_stacked = xy_close & height_valid  # (N,)
        return is_stacked

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
