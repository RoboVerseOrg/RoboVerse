"""Configuration for the Libero kitchen scene8 turn off the stove task."""

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
    "libero_90.kitchen_scene8_turn_off_the_stove",
    "kitchen_scene8_turn_off_the_stove",
)
class LiberoKitchenScene8TurnOffStoveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene8 turn off the stove task.

    Task Description:
    - Turn off the flat stove

    This is a manipulation task that requires:
    1. Locating the stove knob
    2. Rotating or pressing the knob to turn off the stove

    Objects:
    - moka_pot_1 (object): Distractor object (left moka pot)
    - moka_pot_2 (object): Distractor object (right moka pot)
    - flat_stove (fixture): Target stove with knob to turn off

    Goal: (flat_stove turned_off)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects - two moka pots
            RigidObjCfg(
                name="moka_pot_1",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/usd/moka_pot.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/urdf/moka_pot.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="moka_pot_2",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/usd/moka_pot.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/urdf/moka_pot.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            # Fixed fixtures
            ArticulationObjCfg(
                name="flat_stove",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/usd/flat_stove.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/urdf/flat_stove.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/mjcf/flat_stove.xml",
            ),
        ],
        robots=["franka"],
        # Scene configuration
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    # Task parameters
    max_episode_steps = 400  # Moderate complexity task
    task_desc = "Turn off the stove (scene8)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene8_turn_off_the_stove_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: stove is turned off.

        Joint position logic for stove knob (opposite of turn_on):
        - Knob at initial/off position: joint_pos < threshold (e.g., < 0.1)
        - Stove is considered off when knob is rotated back to starting position
        """
        # Get stove knob joint position
        stove_joint_pos = states.objects["flat_stove"].joint_pos  # (N, num_joints)

        off_threshold = 0.1
        is_off = stove_joint_pos[:, 0] < off_threshold  # (N,)
        return is_off

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
