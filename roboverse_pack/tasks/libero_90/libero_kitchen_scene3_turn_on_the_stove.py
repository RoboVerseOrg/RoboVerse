"""Configuration for the Libero kitchen scene3 turn on the stove task."""

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
    "libero_90.kitchen_scene3_turn_on_the_stove",
    "kitchen_scene3_turn_on_the_stove",
)
class LiberoKitchenScene3TurnOnStoveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene3 turn on the stove task.

    This task is transferred from:
    KITCHEN_SCENE3_turn_on_the_stove.bddl

    Task Description:
    - Turn on the flat stove

    This is a manipulation task that requires:
    1. Locating the stove knob
    2. Rotating or pressing the knob to turn on the stove

    Objects from BDDL:
    - flat_stove (fixture): Target stove surface
    - chefmate_8_frypan (object): Distractor object
    - moka_pot (object): Distractor object

    Goal: (flat_stove turned_on)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects
            RigidObjCfg(
                name="moka_pot",
                # usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/usd/moka_pot.usd",
                # urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/urdf/moka_pot.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chefmate_8_frypan",
                # usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/chefmate_8_frypan/usd/chefmate_8_frypan.usd",
                # urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/chefmate_8_frypan/urdf/chefmate_8_frypan.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/mjcf/chefmate_8_frypan.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            # Fixed fixtures
            ArticulationObjCfg(
                name="flat_stove",
                fix_base_link=True,
                # usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/usd/flat_stove.usd",
                # urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/urdf/flat_stove.urdf",
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
    max_episode_steps = 500  # Moderate complexity task
    task_desc = "Turn on the stove (scene3)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene3_turn_on_the_stove_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        # 假设 stove 的开关状态可以通过 states.objects["flat_stove"].joint_state 获取
        # 这里 joint_state[:, knob_index] > threshold 代表已打开
        # knob_index 和 threshold 需根据实际模型调
        knob_index = 0  # 示例：第一个关节为旋钮
        threshold = 0.5  # 示例：大于 0.5 视为打开
        stove_joint_state = states.objects["flat_stove"].joint_pos[:, 0]  # (N,)
        is_on = (stove_joint_state > threshold).all(dim=-1)  # (N,)
        return is_on

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
