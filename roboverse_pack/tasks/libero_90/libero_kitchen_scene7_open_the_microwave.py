"""Configuration for the Libero kitchen scene7 open the microwave task."""

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
    "libero_90.kitchen_scene7_open_the_microwave",
    "kitchen_scene7_open_the_microwave",
)
class LiberoKitchenScene7OpenTheMicrowaveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene7 open the microwave task.

    Task Description:
    - Open the microwave door

    This is a manipulation task that requires:
    1. Grasping and manipulating the microwave door handle
    2. Opening the microwave door to the required position

    Objects:
    - white_bowl (object): Bowl on the table
    - plate (object): Plate on the table
    - microwave (fixture): The microwave with articulated door

    Goal: (Open microwave)
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
    task_desc = "Open the microwave (scene7)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene7_open_the_microwave_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: microwave is open.

        Joint position logic for microwave door:
        - range: [-2.094, 0] (from microwave.xml)
        - closed: joint_pos > -0.01 (near 0)
        - open: joint_pos < -1.5 (sufficiently open)
        """
        microjoint_pos = states.objects["microwave"].joint_pos[:, 0]  # (N,)
        # open joint_pos < -1.5
        opened = microjoint_pos < -1.5
        return opened

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
