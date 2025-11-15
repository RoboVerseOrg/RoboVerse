"""Configuration for the Libero kitchen scene4 put the black bowl on top of the cabinet task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import matrix_from_quat

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet",
    "kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet",
)
class LiberoKitchenScene4PutBlackBowlOnTopOfCabinetTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene4 put the black bowl on top of the cabinet task.

    This task is transferred from:
    KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet.bddl

    Task Description:
    - Put the akita black bowl on top of the white cabinet

    This is a manipulation task that requires:
    1. Grasping akita_black_bowl
    2. Placing it on top of the cabinet

    Objects from BDDL:
    - white_cabinet (fixture): The cabinet with drawers
    - akita_black_bowl(object): Bowl to be placed on the cabinet
    - wine_bottle(object): Distractor object
    - wine_rack(object): Distractor object

    Goal: (On akita_black_bowl white_cabinet_top_surface)
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="akita_black_bowl",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wine_bottle",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/usd/wine_bottle.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/urdf/wine_bottle.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wine_bottle/mjcf/wine_bottle.xml",
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
    task_desc = "Put the black bowl on top of the cabinet (scene4)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_traj_v2.pkl"
    )

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: bowl is on top of the cabinet."""
        bowl_pos = states.objects["akita_black_bowl"].root_state[:, :3]  # (N,3)
        cabinet_pos = states.objects["white_cabinet"].root_state[:, :3]  # (N,3)
        cabinet_quat = states.objects["white_cabinet"].root_state[:, 3:7]  # (N,4)
        # cabinet top bbox参数（需根据实际模型调整）
        bbox_lower = torch.tensor([-0.12, -0.076, 0.22], device=bowl_pos.device)
        bbox_upper = torch.tensor([0.13, 0.11, 0.22 + 0.05], device=bowl_pos.device)
        R = matrix_from_quat(cabinet_quat)  # (N,3,3)
        bowl_local = torch.matmul(R.transpose(1, 2), (bowl_pos - cabinet_pos).unsqueeze(-1)).squeeze(-1)  # (N,3)
        ge_lower = bowl_local >= bbox_lower  # (N,3)
        le_upper = bowl_local <= bbox_upper  # (N,3)
        inside = (ge_lower & le_upper).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
