"""Configuration for the Libero kitchen scene7 put the white bowl to the right of the plate task."""

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
    "libero_90.kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate",
    "kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate",
)
class LiberoKitchenScene7PutTheWhiteBowlToTheRightOfThePlateTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene7 put the white bowl to the right of the plate task.

    Task Description:
    - Put the white bowl to the right of the plate

    This is a manipulation task that requires:
    1. Grasping the white bowl
    2. Placing it to the right side of the plate (positive x direction relative to plate)

    Objects:
    - white_bowl (object): Bowl to be placed to the right of the plate
    - plate (object): Reference object
    - microwave (fixture): The microwave (distractor)

    Goal: (ToTheRightOf white_bowl plate)
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="white_bowl",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/usd/white_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/urdf/white_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml",
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
                name="microwave",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/microwave/usd/microwave.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/microwave/urdf/microwave.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/microwave/mjcf/microwave.xml",
            ),
        ],
        robots=["franka"],
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    max_episode_steps = 300
    task_desc = "Put the white bowl to the right of the plate (scene7)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: white bowl is to the right of the plate.

        Success condition (similar to scene6 mug placement):
        - Bowl is to the right of plate: x_diff in range [0.1, 0.2]
        - Bowl is aligned with plate in y: y_diff in range [-0.08, 0.08]
        - Bowl is on table (similar z height): z_diff in range [-0.02, 0.02]
        """
        bowl_pos = states.objects["white_bowl"].root_state[:, :3]  # (N,3)
        plate_pos = states.objects["plate"].root_state[:, :3]  # (N,3)

        # Calculate relative position: bowl relative to plate
        relative_pos = bowl_pos - plate_pos
        x_diff = relative_pos[:, 0]  # x direction (right is positive)
        y_diff = relative_pos[:, 1]  # y direction
        z_diff = relative_pos[:, 2]  # z direction (height)

        # Check if bowl is to the right of plate x: -0.05 to 0.05; y: 0.05 to 0.15; z: -0.02 to 0.02
        x_in_range = (x_diff > -0.05) & (x_diff < 0.05)
        y_in_range = (y_diff > 0.05) & (y_diff < 0.20)
        z_in_range = (z_diff > -0.02) & (z_diff < 0.02)

        success = x_in_range & y_in_range & z_in_range
        return success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
