"""Configuration for the Libero kitchen scene9 turn on the stove task."""

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
    "libero_90.kitchen_scene9_turn_on_the_stove",
    "kitchen_scene9_turn_on_the_stove",
)
class LiberoKitchenScene9TurnOnStoveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene9 turn on the stove task.

    Task Description:
    - Turn on the flat stove

    This is a manipulation task that requires:
    1. Locating the stove knob
    2. Rotating or pressing the knob to turn on the stove

    Objects:
    - white_bowl (object): Distractor object
    - chefmate_8_frypan (object): Distractor object
    - wooden_two_layer_shelf (fixture): Distractor shelf
    - flat_stove (fixture): Target stove surface

    Goal: (flat_stove turned_on)
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
                name="chefmate_8_frypan",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/usd/chefmate_8_frypan.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/urdf/chefmate_8_frypan.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/mjcf/chefmate_8_frypan.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wooden_two_layer_shelf",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/mjcf/wooden_two_layer_shelf.xml",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/urdf/wooden_two_layer_shelf.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/mjcf/wooden_two_layer_shelf.xml",
            ),
            ArticulationObjCfg(
                name="flat_stove",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/usd/flat_stove.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/urdf/flat_stove.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/mjcf/flat_stove.xml",
            ),
        ],
        robots=["franka"],
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    max_episode_steps = 200
    task_desc = "Turn on the stove (scene9)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene9_turn_on_the_stove_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: stove is on."""
        threshold = 0.5
        stove_joint_state = states.objects["flat_stove"].joint_pos[:, 0]  # (N,)
        is_on = stove_joint_state > threshold  # (N,)
        return is_on

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
