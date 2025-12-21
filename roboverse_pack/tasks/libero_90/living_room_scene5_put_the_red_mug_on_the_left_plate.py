"""Configuration for the Libero living room scene5 put the red mug on the left plate task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene5_put_the_red_mug_on_the_left_plate",
    "living_room_scene5_put_the_red_mug_on_the_left_plate",
)
class LiberoLivingRoomScene5PutTheRedMugOnTheLeftPlateTask(Libero90BaseTask):
    """Configuration for the Libero living room scene5 put the red mug on the left plate task.

    Task Description:
    - Put the red mug on the left plate

    Objects:
    - red_coffee_mug
    - porcelain_mug
    - white_yellow_mug
    - plate_1
    - plate_2

    Goal: red_coffee_mug is on plate_1.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="porcelain_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/usd/porcelain_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/urdf/porcelain_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/mjcf/porcelain_mug.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="red_coffee_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/usd/red_coffee_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/urdf/red_coffee_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/mjcf/red_coffee_mug.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="white_yellow_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/usd/white_yellow_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/urdf/white_yellow_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/mjcf/white_yellow_mug.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="plate_1",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="plate_2",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 200
    task_desc = "Put the red mug on the left plate (living_room_scene5)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_living_room_scene5_put_the_red_mug_on_the_left_plate_traj_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        red_mug_pos = states.objects["red_coffee_mug"].root_state[:, :3]  # (N,3)
        plate_pos = states.objects["plate_1"].root_state[:, :3]  # (N,3)
        # Check if red mug is within a small region above the plate
        range_threshold = 0.06  # Radius of the range in xy plane
        height_threshold = 0.03  # Height threshold above the plate

        # Calculate xy distance between red mug and plate
        xy_distance = torch.norm(red_mug_pos[:, :2] - plate_pos[:, :2], dim=-1)  # (N,)
        # Calculate height difference (red mug z - plate z)
        height_diff = red_mug_pos[:, 2] - plate_pos[:, 2]  # (N,)
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
