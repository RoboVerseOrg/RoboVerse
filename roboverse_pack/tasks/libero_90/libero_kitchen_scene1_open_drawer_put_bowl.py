"""Configuration for the Libero kitchen open drawer and put bowl task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
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
        # Scene configuration (from BDDL problem domain),
    )

    # Task parameters
    max_episode_steps = 500  # Complex task requiring two subtasks
    task_desc = "Open the top drawer of the cabinet and put the bowl in it"

    # Workspace configuration (from BDDL regions)
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_traj_v2.pkl"

    goal = (
        ("Open", "wooden_cabinet", "top_level"),
        ("In", "akita_black_bowl", "wooden_cabinet", "top_region"),
    )

    # env.handler.get_states[0]['objects']['wooden_cabinet']['dof_pos']["top_level"] 获取top

    # rewrite checker
