"""Configuration for the Libero kitchen scene4 put the wine bottle in the bottom drawer of the cabinet task."""

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


def pose_to_mat(pos: torch.Tensor, quat: torch.Tensor | None = None) -> torch.Tensor:
    """Build homogeneous transforms (N,4,4) from batched pos (N,3) and optional quat (N,4)."""
    assert pos.dim() == 2 and pos.shape[-1] == 3, "pos must be (N,3)"
    N = pos.shape[0]
    T = torch.eye(4, dtype=pos.dtype, device=pos.device).unsqueeze(0).repeat(N, 1, 1)
    if quat is None:
        R = torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0).repeat(N, 1, 1)
    else:
        assert quat.shape == (N, 4), "quat must be (N,4)"
        R = matrix_from_quat(quat)  # (N,3,3)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T


@register_task(
    "libero_90.kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    "kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
)
class LiberoKitchenScene4PutWineBottleInBottomDrawerTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene4 put the wine bottle in the bottom drawer of the cabinet task.

    This task is transferred from:
    KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.bddl

    Task Description:
    - Put the wine bottle into the bottom drawer of the white cabinet

    This is a manipulation task that requires:
    1. Picking up the wine bottle from the table
    2. Placing the wine bottle inside the bottom drawer

    Objects from BDDL:
    - white_cabinet (fixture): The cabinet with drawers
    - wine_bottle(object): Bottle to be placed in the drawer
    - akita_black_bowl(object): Distractor object
    - wine_rack(object): Distractor object

    Goal: (In wine_bottle white_cabinet_bottom_region)
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

    max_episode_steps = 500
    task_desc = "Put the wine bottle in the bottom drawer of the cabinet (scene4)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: wine bottle is inside the bottom drawer."""
        bottle_pos = states.objects["wine_bottle"].root_state[:, :3]  # (N,3)
        cabinet_pos = states.objects["white_cabinet"].root_state[:, :3]  # (N,3)
        cabinet_quat = states.objects["white_cabinet"].root_state[:, 3:7]  # (N,4)
        # bottom drawer index assumed to be 0
        bottom_idx = 0
        bbox_displacement = states.objects["white_cabinet"].joint_pos[:, bottom_idx]
        cabinet_T = pose_to_mat(cabinet_pos, cabinet_quat)
        # bottom drawer bbox
        bbox_relative_pos = torch.tensor([0.003, 0.011 + bbox_displacement, 0.065], device=bottle_pos.device)
        bbox_relative_quat = torch.tensor([0.70711, 0.00000, 0.70711, 0.00000], device=bottle_pos.device)
        bbox_relative_T = pose_to_mat(bbox_relative_pos.unsqueeze(0), bbox_relative_quat.unsqueeze(0))
        bbox_half_size = torch.tensor([0.029, 0.075, 0.102], device=bottle_pos.device)
        bbox_T_world = torch.matmul(cabinet_T, bbox_relative_T)
        R = bbox_T_world[:, :3, :3]  # (N,3,3)
        t = bbox_T_world[:, :3, 3]  # (N,3)
        bottle_local = torch.matmul(R.transpose(1, 2), (bottle_pos - t).unsqueeze(-1)).squeeze(-1)  # (N,3)
        eps = 1e-6
        inside = (bottle_local.abs() <= (bbox_half_size.unsqueeze(0) + eps)).all(dim=-1)  # (N,)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
