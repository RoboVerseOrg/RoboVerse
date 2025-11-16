"""Configuration for the Libero kitchen scene6 close the microwave task."""

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
    "libero_90.kitchen_scene6_close_the_microwave",
    "kitchen_scene6_close_the_microwave",
)
class LiberoKitchenScene6CloseTheMicrowaveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene6 close the microwave task."""

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

    max_episode_steps = 600
    task_desc = "Close the microwave (scene6)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene6_close_the_microwave_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: microwave is closed."""
        microjoint_pos = states.objects["microwave"].joint_pos[:, 0]  # (N,)
        closed = microjoint_pos > -0.01
        return closed

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
