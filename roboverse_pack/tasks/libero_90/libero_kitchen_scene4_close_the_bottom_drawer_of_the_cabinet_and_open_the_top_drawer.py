"""Configuration for the Libero kitchen scene2 stack the middle black bowl on the back black bowl task."""

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
    "libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
    "kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
)
class LiberoKitchenScene4CloseTheBottomDrawerOfTheCabinetAndOpenTheTopDrawerTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene4 close the bottom drawer of the cabinet and open the top drawer task.

    This task is transferred from:
    KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl

    Task Description:
    - Open the top drawer of the cabinet and close the bottom drawer.

    This is a manipulation task that requires:
    1. Opening the top drawer
    2. Closing the bottom drawer

    Objects from BDDL:
    - white_cabinet (fixture): The cabinet with drawers
    - akita_black_bowl(object): One bowls on the table
    - wine_bottle(object): Distractor object
    - wine_rack(object): Distractor object

    Goal: (Open white_cabinet_top_region) AND (Closed white_cabinet_bottom_region)
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
    task_desc = "Close the bottom drawer of the cabinet and open the top drawer (scene4)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0.0, 0, 0.90),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_traj_v2.pkl"

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success checker: top drawer open AND bottom drawer closed."""
        # Get the white cabinet joint positions
        cabinet_joint_pos = states.objects["white_cabinet"].joint_pos  # (N, num_joints)

        # Joint position logic:
        # negative value (e.g., -0.1580) = drawer open
        # positive/zero value = drawer closed
        bottom_threshold = -0.01  # Bottom drawer closed: position > this
        top_threshold = -0.13  # Top drawer open: position < this

        # Check bottom drawer is closed (joint_pos > threshold, close to 0 or positive)
        bottom_closed = cabinet_joint_pos[:, 0] > bottom_threshold  # (N,)

        # Check top drawer is open (joint_pos < threshold, negative)
        top_open = cabinet_joint_pos[:, 2] < top_threshold  # (N,)
        # Both conditions must be satisfied
        is_success = bottom_closed & top_open  # (N,)
        return is_success

    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(Libero90BaseTask, self).reset(states, env_ids)
        return states
