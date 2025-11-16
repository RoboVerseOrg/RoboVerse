"""Configuration for the Libero kitchen scene5 put the ketchup in the top drawer of the cabinet task."""

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
        R = matrix_from_quat(quat)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T


@register_task(
    "libero_90.kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    "kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
)
class LiberoKitchenScene5PutKetchupInTopDrawerTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene5 put the ketchup in the top drawer of the cabinet task."""

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
                name="plate",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="ketchup",
                fix_base_link=False,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
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
    task_desc = "Put the ketchup in the top drawer of the cabinet (scene5)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        ketchup_pos = states.objects["ketchup"].root_state[:, :3]
        cabinet_pos = states.objects["white_cabinet"].root_state[:, :3]
        cabinet_quat = states.objects["white_cabinet"].root_state[:, 3:7]
        top_idx = 2  # top drawer index
        bbox_displacement = states.objects["white_cabinet"].joint_pos[:, top_idx]
        cabinet_T = pose_to_mat(cabinet_pos, cabinet_quat)
        bbox_relative_pos = torch.tensor([0.00328, 0.01128 + bbox_displacement, 0.18563], device=ketchup_pos.device)
        bbox_relative_quat = torch.tensor([0.70711, 0.00000, 0.70711, 0.00000], device=ketchup_pos.device)
        bbox_relative_T = pose_to_mat(bbox_relative_pos.unsqueeze(0), bbox_relative_quat.unsqueeze(0))
        bbox_half_size = torch.tensor([0.02993, 0.07561, 0.10224], device=ketchup_pos.device)
        bbox_T_world = torch.matmul(cabinet_T, bbox_relative_T)
        R = bbox_T_world[:, :3, :3]
        t = bbox_T_world[:, :3, 3]
        ketchup_local = torch.matmul(R.transpose(1, 2), (ketchup_pos - t).unsqueeze(-1)).squeeze(-1)
        eps = 1e-6
        inside = (ketchup_local.abs() <= (bbox_half_size.unsqueeze(0) + eps)).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
