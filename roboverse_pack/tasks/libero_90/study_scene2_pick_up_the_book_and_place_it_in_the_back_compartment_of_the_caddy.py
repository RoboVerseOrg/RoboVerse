"""Configuration for the Libero study scene2 pick up the book and place it in the back compartment of the caddy task."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.types import TensorState

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.study_scene2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "study_scene2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
)
class LiberoStudyScene2PickUpTheBookAndPlaceInTheBackCompartmentOfTheCaddyTask(Libero90BaseTask):
    """Configuration for the Libero study scene2 pick up the book and place it in the back compartment of the caddy task.

    Task Description:
    - Pick up the black_book from the table
    - Place the black_book inside the desk_caddy/back_contain_region

    This is a manipulation task that requires:
    1. Picking up the black_book from the table
    2. Placing the black_book inside the desk_caddy/back_contain_region

    Objects:
    - black_book (target)
    - red_coffee_mug
    - desk_caddy (goal container)

    Goal: Place black_book inside desk_caddy/back_contain_region.
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
                name="red_coffee_mug",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/usd/red_coffee_mug.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/urdf/red_coffee_mug.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/red_coffee_mug/mjcf/red_coffee_mug.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="desk_caddy",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/usd/desk_caddy.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/urdf/desk_caddy.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/desk_caddy/mjcf/desk_caddy.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 250
    task_desc = "Pick up the black_book and place it in the back compartment of the desk_caddy (study_scene2)"

    workspace_name = ("study_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_study_scene2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: black_book is inside desk_caddy/back_contain_region bounding box."""
        book_pos = states.objects["black_book"].root_state[:, :3]
        N = book_pos.shape[0]
        region_mat = self.handler.physics.named.data.site_xmat["desk_caddy/back_contain_region"]
        region_pos = self.handler.physics.named.data.site_xpos["desk_caddy/back_contain_region"]
        region_R = torch.from_numpy(region_mat).float().reshape(3, 3).unsqueeze(0).expand(N, 3, 3).to(book_pos.device)
        region_t = torch.from_numpy(region_pos).float().unsqueeze(0).expand(N, 3).to(book_pos.device)
        half_size = torch.tensor([0.02775, 0.06216, 0.06046], device=book_pos.device)
        book_local = torch.matmul(region_R.transpose(1, 2), (book_pos - region_t).unsqueeze(-1)).squeeze(-1)
        inside = (book_local.abs() <= (half_size + 1e-6)).all(dim=-1)
        return inside

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
