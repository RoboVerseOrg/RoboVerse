"""Configuration for the Libero kitchen scene4 put the wine bottle on the wine rack task."""

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
    "libero_90.kitchen_scene4_put_the_wine_bottle_on_the_wine_rack",
    "kitchen_scene4_put_the_wine_bottle_on_the_wine_rack",
)
class LiberoKitchenScene4PutWineBottleOnWineRackTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene4 put the wine bottle on the wine rack task.

    This task is transferred from:
    KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl

    Task Description:
    - Put the wine bottle on the wine rack

    This is a manipulation task that requires:
    1. Grasping wine_bottle
    2. Placing it on the wine rack

    Objects from BDDL:
    - white_cabinet (fixture): The cabinet with drawers
    - wine_bottle(object): Bottle to be placed on the rack
    - akita_black_bowl(object): Distractor object
    - wine_rack(object): Target rack

    Goal: (On wine_bottle wine_rack_top_surface)
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="wine_bottle",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/usd/wine_bottle.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/urdf/wine_bottle.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/mjcf/wine_bottle.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wine_rack",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_rack/usd/wine_rack.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_rack/urdf/wine_rack.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_rack/mjcf/wine_rack.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="white_cabinet",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/white_cabinet/usd/white_cabinet.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/white_cabinet/urdf/white_cabinet.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/white_cabinet/mjcf/white_cabinet.xml",
            ),
        ],
        robots=["franka"],
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    max_episode_steps = 300
    task_desc = "Put the wine bottle on the wine rack (scene4)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_traj_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: wine bottle is on top of the wine rack."""
        bottle_pos = states.objects["wine_bottle"].root_state[:, :3]  # (N,3)
        N = bottle_pos.shape[0]

        # Get rack top pose and expand to N environments
        rack_top_mat = self.handler.physics.named.data.site_xmat["wine_rack_stand/top_region"]  # (9,)
        rack_top_pos = self.handler.physics.named.data.site_xpos["wine_rack_stand/top_region"]  # (3,)

        # Expand to (N,3,3) and (N,3)
        rack_top_R = (
            torch.from_numpy(rack_top_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(bottle_pos.device)
        )  # (N,3,3)
        rack_top_t = torch.from_numpy(rack_top_pos).float().unsqueeze(0).expand(N, 3).to(bottle_pos.device)  # (N,3)

        # wine rack top region (size found in wine_rack.xml)
        bbox_lower = torch.tensor([-0.4, -0.07894, -0.13438], device=bottle_pos.device)
        bbox_upper = torch.tensor([0.00, 0.07894, 0.13438], device=bottle_pos.device)

        # Transform bottle position to rack local frame
        bottle_local = torch.matmul(rack_top_R.transpose(1, 2), (bottle_pos - rack_top_t).unsqueeze(-1)).squeeze(
            -1
        )  # (N,3)
        ge_lower = bottle_local >= bbox_lower  # (N,3)
        le_upper = bottle_local <= bbox_upper  # (N,3)
        inside = (ge_lower & le_upper).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
