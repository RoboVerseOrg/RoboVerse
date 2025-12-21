"""Configuration for the Libero study scene4 pick up the book in the middle and place it on the cabinet shelf task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.study_scene4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    "study_scene4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
)
class LiberoStudyScene4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelfTask(Libero90BaseTask):
    """Configuration for the Libero study scene4 pick up the book in the middle and place it on the cabinet shelf task.

    Task Description:
    - Pick up the black_book (middle)
    - Place it on the wooden_two_layer_shelf (cabinet shelf)

    Objects:
    - black_book (target)
    - yellow_book_1
    - yellow_book_2
    - wooden_two_layer_shelf (goal shelf)

    Goal: Place black_book on wooden_two_layer_shelf (cabinet shelf).
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="black_book",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/usd/black_book.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/urdf/black_book.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/black_book/mjcf/black_book.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="yellow_book_1",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/usd/yellow_book.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/urdf/yellow_book.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/mjcf/yellow_book.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="yellow_book_2",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/usd/yellow_book.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/urdf/yellow_book.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/yellow_book/mjcf/yellow_book.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wooden_two_layer_shelf",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/usd/wooden_two_layer_shelf.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/urdf/wooden_two_layer_shelf.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/mjcf/wooden_two_layer_shelf.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 250
    task_desc = "Pick up the black_book in the middle and place it on the wooden_two_layer_shelf (study_scene4)"

    workspace_name = ("study_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_study_scene4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: frying pan is in the bottom region of the cabinet shelf.

        Success condition (similar to wine_rack in kitchen4):
        - Frying pan is within the bounding box region defined by the bottom_region site
        - Uses the site's position and orientation from physics data
        """
        book_pos = states.objects["black_book"].root_state[:, :3]  # (N,3)
        N = book_pos.shape[0]

        # Get shelf top_side site pose and expand to N environments
        shelf_top_mat = self.handler.physics.named.data.site_xmat["wooden_two_layer_shelf/top_side"]  # (9,)
        shelf_top_pos = self.handler.physics.named.data.site_xpos["wooden_two_layer_shelf/top_side"]  # (3,)

        shelf_top_R = (
            torch.from_numpy(shelf_top_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(book_pos.device)
        )  # (N,3,3)
        shelf_top_t = torch.from_numpy(shelf_top_pos).float().unsqueeze(0).expand(N, 3).to(book_pos.device)  # (N,3)

        # top_side site half-size from wooden_two_layer_shelf.xml: size="0.03272 0.05000 0.11027"
        bbox_lower = torch.tensor([-0.03272, -0.05000, -0.11027], device=book_pos.device)
        bbox_upper = torch.tensor([0.03272, 0.05000, 0.11027], device=book_pos.device)

        # Transform book position to shelf top_side local frame
        book_local = torch.matmul(shelf_top_R.transpose(1, 2), (book_pos - shelf_top_t).unsqueeze(-1)).squeeze(
            -1
        )  # (N,3)
        ge_lower = book_local >= bbox_lower  # (N,3)
        le_upper = book_local <= bbox_upper  # (N,3)
        inside = (ge_lower & le_upper).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
