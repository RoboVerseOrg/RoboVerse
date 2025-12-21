"""Configuration for the Libero living room scene6 put the chocolate pudding to the left of the plate task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "living_room_scene6_put_the_chocolate_pudding_to_the_left_of_the_plate",
)
class LiberoLivingRoomScene6PutTheChocolatePuddingToTheLeftOfPlateTask(Libero90BaseTask):
    """Task: Put the chocolate_pudding to the left of the plate in living_room_scene6.

    Objects:
    - porcelain_mug
    - red_coffee_mug
    - plate (reference)
    - chocolate_pudding (target)

    Goal: Place chocolate_pudding to the left of the plate.
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
                name="plate",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chocolate_pudding",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/usd/chocolate_pudding.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/urdf/chocolate_pudding.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/mjcf/chocolate_pudding.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 250
    task_desc = "Put the chocolate_pudding to the left of the plate (living_room_scene6)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0.42),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene6_put_the_chocolate_pudding_to_the_left_of_the_plate_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success: chocolate_pudding is to the left of the plate (x < plate.x - threshold)."""
        chocolate_pos = states.objects["chocolate_pudding"].root_state[:, :3]
        plate_pos = states.objects["plate"].root_state[:, :3]

        # Calculate relative position: chocolate relative to plate
        relative_pos = chocolate_pos - plate_pos
        x_diff = relative_pos[:, 0]  # x direction
        y_diff = relative_pos[:, 1]  # y direction

        # Check if chocolate pudding is in the target region relative to plate, range determined by checking trajectory data
        # x: -0.05 to 0.05, y: 0.0 to 0.2
        x_in_range = (x_diff > -0.05) & (x_diff < 0.05)
        y_in_range = (y_diff > 0) & (y_diff < 0.2)

        success = x_in_range & y_in_range
        return success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
