"""Configuration for the Libero kitchen scene9 put the frying pan on the cabinet shelf task."""

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
    "libero_90.kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf",
    "kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf",
)
class LiberoKitchenScene9PutTheFryingPanOnTheCabinetShelfTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene9 put the frying pan on the cabinet shelf task.

    Task Description:
    - Put the chefmate 8 frying pan on the wooden two layer shelf

    This is a manipulation task that requires:
    1. Grasping the chefmate_8_frypan
    2. Placing it on top of the wooden_two_layer_shelf

    Objects:
    - white_bowl (object): Distractor object
    - chefmate_8_frypan (object): Frying pan to be placed on the shelf
    - wooden_two_layer_shelf (fixture): Target shelf with two layers
    - flat_stove (fixture): Distractor stove

    Goal: (On chefmate_8_frypan wooden_two_layer_shelf)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects
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
            # Fixed fixtures
            RigidObjCfg(
                name="wooden_two_layer_shelf",
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
        # Scene configuration
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    # Task parameters
    max_episode_steps = 300  # Moderate to high complexity task
    task_desc = "Put the frying pan on the cabinet shelf (scene9)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_traj_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: frying pan is on the cabinet shelf top region.

        Success condition (similar to wine_rack in kitchen4):
        - Frying pan is within the bounding box region defined by the top_region site
        - Uses the site's position and orientation from physics data
        """
        frypan_pos = states.objects["chefmate_8_frypan"].root_state[:, :3]  # (N,3)
        N = frypan_pos.shape[0]

        # Get shelf top_region site pose and expand to N environments
        shelf_top_mat = self.handler.physics.named.data.site_xmat["wooden_two_layer_shelf/top_region"]  # (9,)
        shelf_top_pos = self.handler.physics.named.data.site_xpos["wooden_two_layer_shelf/top_region"]  # (3,)

        # Expand to (N,3,3) and (N,3)
        shelf_top_R = (
            torch.from_numpy(shelf_top_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(frypan_pos.device)
        )  # (N,3,3)
        shelf_top_t = torch.from_numpy(shelf_top_pos).float().unsqueeze(0).expand(N, 3).to(frypan_pos.device)  # (N,3)

        # top_region site half-size from wooden_two_layer_shelf.xml: size="0.03272 0.05000 0.11027"
        bbox_lower = torch.tensor([-0.03272, -0.05000, -0.11027], device=frypan_pos.device)
        bbox_upper = torch.tensor([0.03272, 0.05000, 0.11027], device=frypan_pos.device)

        # Transform frying pan position to shelf top_region local frame
        frypan_local = torch.matmul(shelf_top_R.transpose(1, 2), (frypan_pos - shelf_top_t).unsqueeze(-1)).squeeze(
            -1
        )  # (N,3)
        ge_lower = frypan_local >= bbox_lower  # (N,3)
        le_upper = frypan_local <= bbox_upper  # (N,3)
        inside = (ge_lower & le_upper).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
