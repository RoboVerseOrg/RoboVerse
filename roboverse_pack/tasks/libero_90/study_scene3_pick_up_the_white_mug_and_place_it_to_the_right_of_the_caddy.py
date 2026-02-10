"""Configuration for the Libero study scene3 pick up the white mug and place it to the right of the caddy task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.study_scene3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    "study_scene3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
)
class LiberoStudyScene3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddyTask(Libero90BaseTask):
    """Configuration for the Libero study scene3 pick up the white mug and place it to the right of the caddy task.

    Task Description:
    - Pick up the porcelain_mug from the table
    - Place the porcelain_mug inside the desk_caddy/right_contain_region

    This is a manipulation task that requires:
    1. Picking up the porcelain_mug from the table
    2. Placing the porcelain_mug inside the desk_caddy/right_contain_region

    Objects:
    - black_book
    - red_coffee_mug
    - porcelain_mug (target)
    - desk_caddy (goal container)

    Goal: Place porcelain_mug inside desk_caddy/right_contain_region.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="black_book",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/usd/black_book.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/urdf/black_book.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/mjcf/black_book.xml",
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
                name="desk_caddy",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/usd/desk_caddy.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/urdf/desk_caddy.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/mjcf/desk_caddy.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="porcelain_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/usd/porcelain_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/urdf/porcelain_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/mjcf/porcelain_mug.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 200
    task_desc = "Pick up the porcelain_mug and place it to the right of the desk_caddy (study_scene3)"

    workspace_name = ("study_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_study_scene3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: porcelain_mug is to the right of desk_caddy (x > caddy.x + threshold, y/z close)."""
        mug_pos = states.objects["porcelain_mug"].root_state[:, :3]
        caddy_pos = states.objects["desk_caddy"].root_state[:, :3]
        # Calculate relative position: chocolate relative to plate
        relative_pos = mug_pos - caddy_pos
        x_diff = relative_pos[:, 0]  # x direction
        y_diff = relative_pos[:, 1]  # y direction

        # Check if chocolate pudding is in the target region relative to plate, range determined by checking trajectory data
        # x: -0.05 to 0.05, y: 0.0 to 0.2, z: 0.0 to 0.05
        x_in_range = (x_diff > -0.15) & (x_diff < 0.15)
        y_in_range = (y_diff > 0.2) & (y_diff < 0.4)
        z_in_range = (relative_pos[:, 2] > 0.0) & (relative_pos[:, 2] < 0.05)
        success = x_in_range & y_in_range & z_in_range
        return success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
