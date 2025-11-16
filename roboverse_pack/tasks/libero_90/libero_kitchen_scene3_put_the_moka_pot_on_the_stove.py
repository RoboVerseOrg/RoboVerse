"""Configuration for the Libero kitchen scene3 put the moka pot on the stove task."""

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
    "libero_90.kitchen_scene3_put_the_moka_pot_on_the_stove",
    "kitchen_scene3_put_the_moka_pot_on_the_stove",
)
class LiberoKitchenScene3PutMokaPotOnStoveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene3 put the moka pot on the stove task.

    This task is transferred from:
    KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl

    Task Description:
    - Put the moka pot on the flat stove

    This is a manipulation task that requires:
    1. Grasping the moka_pot
    2. Placing it on top of the flat_stove

    Objects from BDDL:
    - moka_pot (object): Moka pot to be placed on the stove
    - chefmate_8_frypan (object): Distractor object
    - flat_stove (fixture): Target stove surface

    Goal: (On moka_pot flat_stove)
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
    max_episode_steps = 300  # Moderate complexity task
    task_desc = "Put the moka pot on the stove (scene3)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene3_put_the_moka_pot_on_the_stove_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        moka_pot_pos = states.objects["moka_pot"].root_state[:, :3]  # (N,3)
        stove_center_single = self.handler.physics.named.data.site_xpos["flat_stove/burner"]  # (3,)
        N = moka_pot_pos.shape[0]
        stove_center_pos = (
            torch.tensor(stove_center_single, device=moka_pot_pos.device).reshape(1, 3).repeat(N, 1)
        )  # (N,3)

        # Check if moka pot is within a region above the stove
        range_threshold = 0.06  # Radius of the range in xy plane (smaller for moka pot)
        height_threshold = 0.05  # Height threshold above the stove

        # Calculate xy distance between moka pot and stove
        xy_distance = torch.norm(moka_pot_pos[:, :2] - stove_center_pos[:, :2], dim=-1)  # (N,)
        # Calculate height difference (moka pot z - stove z)
        height_diff = moka_pot_pos[:, 2] - stove_center_pos[:, 2]  # (N,)

        # Check both conditions: xy distance < range AND 0 < height_diff < height_threshold
        xy_close = xy_distance < range_threshold  # (N,)
        height_valid = (height_diff > 0) & (height_diff < height_threshold)  # (N,)
        is_on_stove = xy_close & height_valid  # (N,)

        return is_on_stove

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
