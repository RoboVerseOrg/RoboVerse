"""Configuration for the Libero kitchen open drawer and put bowl task."""

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


@register_task("libero_90.kitchen_scene1_open_drawer_put_bowl", "kitchen_open_drawer_put_bowl")
class LiberoKitchenOpenDrawerPutBowlTask(Libero90BaseTask):
    """Configuration for the Libero kitchen open drawer and put bowl task.

    This task is transferred from:
    KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl

    Task Description:
    - Open the top drawer of the wooden cabinet
    - Put the akita black bowl into the top drawer

    This is a complex manipulation task that requires:
    1. Grasping and manipulating the cabinet drawer (articulated object)
    2. Opening the top drawer to the required position
    3. Picking up the bowl from the table
    4. Placing the bowl inside the opened drawer

    Objects from BDDL:
    - wooden_cabinet_1 (fixture): The cabinet with drawers
    - akita_black_bowl_1 (object): Bowl to be placed in the drawer
    - plate_1 (object): Plate on the table (distractor)

    Goal: (And (Open wooden_cabinet_1_top_region) (In akita_black_bowl_1 wooden_cabinet_1_top_region))
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects (from BDDL :objects section)
            RigidObjCfg(
                name="akita_black_bowl",  # 碗的位置似乎不对，在libero里换角度观察
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
            ),
            RigidObjCfg(
                name="plate",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/usd/plate.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/urdf/plate.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/plate/mjcf/plate.xml",
            ),
            # Fixed fixtures (from BDDL :fixtures section)
            # Note: kitchen_table is handled by the scene/arena
            ArticulationObjCfg(
                name="wooden_cabinet",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/usd/wooden_cabinet.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/urdf/wooden_cabinet.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/wooden_cabinet/mjcf/wooden_cabinet.xml",
            ),
        ],
        robots=["franka"],
        # Scene configuration (from BDDL problem domain)
        scene=SceneCfg(
            name="libero_kitchen_tabletop",
            mjcf_path="roboverse_data/assets/libero/scenes/libero_tabletop_base_style.xml",
        ),
    )

    # Task parameters
    max_episode_steps = 500  # Complex task requiring two subtasks
    task_desc = "Open the top drawer of the cabinet and put the bowl in it"

    # Workspace configuration (from BDDL regions)
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.91),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_traj_v2.pkl"

    # env.handler.get_states[0]['objects']['wooden_cabinet']['dof_pos']["top_level"] 获取top
    def _terminated(self, states: TensorState) -> torch.Tensor:  # 需要仔细检查一下
        """Task success checker."""
        # get bowl/cabinet poses
        bowl_pos = states.objects["akita_black_bowl"].root_state[:, :3]  # (N,3)
        cabinet_pos = states.objects["wooden_cabinet"].root_state[:, :3]  # (N,3)
        cabinet_quat = states.objects["wooden_cabinet"].root_state[:, 3:7]  # (N,4)
        top_idx = self.handler.get_joint_names("wooden_cabinet").index("top_level")
        bbox_displacement = self.handler.get_states(mode="tensor").objects["wooden_cabinet"].joint_pos[:, top_idx]
        # Build homogeneous transforms (N,4,4)
        # bowl_T = pose_to_mat(bowl_pos, None)              # identity rotation
        cabinet_T = pose_to_mat(cabinet_pos, cabinet_quat)
        # get top drawer T
        bbox_relative_pos = torch.tensor(
            [0.00328, 0.01128 + bbox_displacement, 0.18563], device=bowl_pos.device
        )  # relative to cabinet frame
        bbox_relative_quat = torch.tensor([0.70711, 0.00000, 0.70711, 0.00000], device=bowl_pos.device)
        bbox_relative_T = pose_to_mat(bbox_relative_pos.unsqueeze(0), bbox_relative_quat.unsqueeze(0))  # (1,4,4)
        bbox_half_size = torch.tensor([0.02993, 0.07561, 0.10224], device=bowl_pos.device)  # (3,) 需要进一步确认
        # Compose with cabinet_T to get bbox transform in world coordinates (N,4,4)
        bbox_T_world = torch.matmul(cabinet_T, bbox_relative_T)
        # Transform bowl position into bbox local frame and test inclusion
        R = bbox_T_world[:, :3, :3]  # (N,3,3)
        t = bbox_T_world[:, :3, 3]  # (N,3)
        bowl_local = torch.matmul(R.transpose(1, 2), (bowl_pos - t).unsqueeze(-1)).squeeze(-1)  # (N,3)
        eps = 1e-6
        inside = (bowl_local.abs() <= (bbox_half_size.unsqueeze(0) + eps)).all(dim=-1)  # (N,)

        return inside

    # rewrite checker
    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
