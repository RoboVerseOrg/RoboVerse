"""Configuration for the Libero kitchen scene2 put the middle black bowl on top of the cabinet task."""

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
    """Build homogeneous transforms (N,4,4) from batched pos (N,3) and optional quat (N,4).

    If quat is None, rotation is identity for all N.
    """
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
    "libero_90.kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet",
)
class LiberoKitchenScene2PutMiddleBowlOnCabinetTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene2 put the middle black bowl on top of the cabinet task.

    This task is transferred from:
    KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet.bddl

    Task Description:
    - Put the akita black bowl in the middle (bowl_2) on top of the cabinet

    This is a manipulation task that requires:
    1. Grasping akita_black_bowl_2
    2. Placing it on top of the cabinet

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1/2/3 (object): Three bowls on the table
    - plate_1 (object): Distractor plate

    Goal: (On akita_black_bowl_2 wooden_cabinet_1_top_surface)
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="akita_black_bowl_1",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl_2",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl_3",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="plate",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="wooden_cabinet",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/usd/wooden_cabinet.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/urdf/wooden_cabinet.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/mjcf/wooden_cabinet.xml",
            ),
        ],
        robots=["franka"],
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    max_episode_steps = 300
    task_desc = "Put the middle black bowl (bowl_2) on top of the cabinet (scene2)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker."""
        bowl_pos = states.objects["akita_black_bowl_2"].root_state[:, :3]  # (N,3)
        cabinet_pos = states.objects["wooden_cabinet"].root_state[:, :3]  # (N,3)
        cabinet_quat = states.objects["wooden_cabinet"].root_state[:, 3:7]  # (N,4)
        bbox_lower = torch.tensor([-0.12, -0.076, 0.22], device=bowl_pos.device)  # x-, y-, z-
        bbox_upper = torch.tensor([0.13, 0.11, 0.22 + 0.05], device=bowl_pos.device)  # x+, y+, z+
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
