"""Configuration for the Libero kitchen scene7 put the white bowl on the plate task."""

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
    "libero_90.kitchen_scene7_put_the_white_bowl_on_the_plate",
    "kitchen_scene7_put_the_white_bowl_on_the_plate",
)
class LiberoKitchenScene7PutTheWhiteBowlOnThePlateTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene7 put the white bowl on the plate task.

    Task Description:
    - Put the white bowl on the plate

    This is a manipulation task that requires:
    1. Grasping the white bowl
    2. Placing it on top of the plate

    Objects:
    - white_bowl (object): Bowl to be placed on the plate
    - plate (object): Target plate
    - microwave (fixture): The microwave (distractor)

    Goal: (On white_bowl plate)
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
    task_desc = "Put the white bowl on the plate (scene7)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene7_put_the_white_bowl_on_the_plate_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: white bowl is on the plate.

        Success condition:
        - Bowl is within xy range of plate center (< 0.06m radius)
        - Bowl is slightly above plate (0 < height_diff < 0.03m)
        """
        bowl_pos = states.objects["white_bowl"].root_state[:, :3]  # (N,3)
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

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
