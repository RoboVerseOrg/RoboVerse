"""Configuration for the Libero kitchen scene3 turn on the stove and put the frying pan on it task."""

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
    "libero_90.kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it",
)
class LiberoKitchenScene3TurnOnStoveAndPutFryingPanTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene3 turn on the stove and put the frying pan on it task.

    This task is transferred from:
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl

    Task Description:
    - Turn on the flat stove
    - Put the chefmate 8" frying pan on the flat stove

    This is a compound manipulation task that requires:
    1. Locating and turning on the stove knob
    2. Grasping the chefmate_8_frypan
    3. Placing it on top of the flat_stove

    Objects from BDDL:
    - chefmate_8_frypan (object): Frying pan to be placed on the stove
    - moka_pot (object): Distractor object
    - flat_stove (fixture): Target stove surface

    Goal: (flat_stove turned_on) AND (On chefmate_8_frypan flat_stove)
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
    max_episode_steps = 500  # Compound task, may need more steps
    task_desc = "Turn on the stove and put the frying pan on it (scene3)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: stove is on AND frying pan is on stove."""
        # 1.  check if the stove is turned on
        threshold = 0.5
        stove_joint_state = states.objects["flat_stove"].joint_pos[:, 0]  # (N,)
        stove_on = stove_joint_state > threshold  # (N,)

        # 2. check if the frying pan is on the stove
        frypan_pos = states.objects["chefmate_8_frypan"].root_state[:, :3]  # (N,3)
        stove_center_single = self.handler.physics.named.data.site_xpos["flat_stove/burner"]  # (3,)
        N = frypan_pos.shape[0]
        stove_center_pos = (
            torch.tensor(stove_center_single, device=frypan_pos.device).reshape(1, 3).repeat(N, 1)
        )  # (N,3)
        range_threshold = 0.08  # Radius of the range in xy plane
        height_threshold = 0.03  # Height threshold above the stove
        xy_distance = torch.norm(frypan_pos[:, :2] - stove_center_pos[:, :2], dim=-1)  # (N,)
        height_diff = frypan_pos[:, 2] - stove_center_pos[:, 2]  # (N,)
        xy_close = xy_distance < range_threshold  # (N,)
        height_valid = (height_diff > 0) & (height_diff < height_threshold)  # (N,)
        frypan_on_stove = xy_close & height_valid  # (N,)

        # Final success condition
        is_success = stove_on & frypan_on_stove  # (N,)
        return is_success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
