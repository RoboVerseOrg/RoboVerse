"""Configuration for the Libero kitchen scene6 put the yellow and white mug to the front of the white mug task."""

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
    "libero_90.kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    "kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
)
class LiberoKitchenScene6PutYellowWhiteMugFrontWhiteMugTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene6 put the yellow and white mug to the front of the white mug task."""

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
                name="white_yellow_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/usd/white_yellow_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/urdf/white_yellow_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/white_yellow_mug/mjcf/white_yellow_mug.xml",
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

    max_episode_steps = 200
    task_desc = "Put the yellow and white mug to the front of the white mug (scene6)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: yellow and white mug is in front of the white mug."""
        yellow_pos = states.objects["white_yellow_mug"].root_state[:, :3]
        white_pos = states.objects["porcelain_mug"].root_state[:, :3]

        # Calculate relative position: yellow relative to white
        relative_pos = yellow_pos - white_pos
        x_diff = relative_pos[:, 0]  # x direction
        y_diff = relative_pos[:, 1]  # y direction

        # Check if yellow mug is in the target region relative to white mug, range determined by checking trajectory data
        # x: 0.0 to 0.2, y: -0.05 to +0.05
        x_in_range = (x_diff > 0.0) & (x_diff < 0.2)
        y_in_range = (y_diff > -0.05) & (y_diff < 0.05)

        success = x_in_range & y_in_range
        return success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
