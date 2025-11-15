"""Configuration for the Libero kitchen scene9 put the white bowl on top of the cabinet task."""

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
    "libero_90.kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet",
    "kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet",
)
class LiberoKitchenScene9PutTheWhiteBowlOnTopOfTheCabinetTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene9 put the white bowl on top of the cabinet task.

    Task Description:
    - Put the white bowl on top of the wooden two layer shelf

    This is a manipulation task that requires:
    1. Grasping the white_bowl
    2. Placing it on top of the wooden_two_layer_shelf (top_side surface)

    Objects:
    - white_bowl (object): Bowl to be placed on top of the cabinet
    - chefmate_8_frypan (object): Distractor object
    - wooden_two_layer_shelf (fixture): Target shelf with two layers
    - flat_stove (fixture): Distractor stove

    Goal: (On white_bowl wooden_two_layer_shelf_top_side)
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
    max_episode_steps = 300
    task_desc = "Put the white bowl on top of the cabinet (scene9)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    # Trajectory file path
    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_traj_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: white bowl is on top of the cabinet (top_side surface).

        Success condition (similar to wine_rack in kitchen4):
        - White bowl is within the bounding box region defined by the top_side site
        - Uses the site's position and orientation from physics data
        """
        bowl_pos = states.objects["white_bowl"].root_state[:, :3]  # (N,3)
        N = bowl_pos.shape[0]

        # Get shelf top_side site pose and expand to N environments
        shelf_top_mat = self.handler.physics.named.data.site_xmat["wooden_two_layer_shelf/top_side"]  # (9,)
        shelf_top_pos = self.handler.physics.named.data.site_xpos["wooden_two_layer_shelf/top_side"]  # (3,)

        shelf_top_R = (
            torch.from_numpy(shelf_top_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(bowl_pos.device)
        )  # (N,3,3)
        shelf_top_t = torch.from_numpy(shelf_top_pos).float().unsqueeze(0).expand(N, 3).to(bowl_pos.device)  # (N,3)

        # top_side site half-size from wooden_two_layer_shelf.xml: size="0.12534 0.09438 0.00147"
        bbox_lower = torch.tensor([-0.12534, -0.09438, 0], device=bowl_pos.device)
        bbox_upper = torch.tensor([0.12534, 0.09438, 0.03], device=bowl_pos.device)

        # Transform white bowl position to shelf top_side local frame
        bowl_local = torch.matmul(shelf_top_R.transpose(1, 2), (bowl_pos - shelf_top_t).unsqueeze(-1)).squeeze(
            -1
        )  # (N,3)
        ge_lower = bowl_local >= bbox_lower  # (N,3)
        le_upper = bowl_local <= bbox_upper  # (N,3)
        inside = (ge_lower & le_upper).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
