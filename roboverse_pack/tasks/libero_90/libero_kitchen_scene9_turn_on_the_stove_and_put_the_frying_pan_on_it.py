"""Configuration for the Libero kitchen scene9 turn on the stove and put the frying pan on it task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it",
)
class LiberoKitchenScene9TurnOnStoveAndPutFryingPanTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene9 turn on the stove and put the frying pan on it task.

    Task Description:
    - Turn on the flat stove
    - Put the chefmate 8 frying pan on the flat stove

    This is a compound manipulation task that requires:
    1. Locating and turning on the stove knob
    2. Grasping the chefmate_8_frypan
    3. Placing it on top of the flat_stove

    Objects:
    - white_bowl (object): Distractor object
    - chefmate_8_frypan (object): Frying pan to be placed on the stove
    - wooden_two_layer_shelf (fixture): Distractor shelf
    - flat_stove (fixture): Target stove surface

    Goal: (flat_stove turned_on) AND (On chefmate_8_frypan flat_stove)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects
            RigidObjCfg(
                name="white_bowl",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/usd/white_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/urdf/white_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chefmate_8_frypan",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/usd/chefmate_8_frypan.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/urdf/chefmate_8_frypan.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/chefmate_8_frypan/mjcf/chefmate_8_frypan.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            # Fixed fixtures
            RigidObjCfg(
                name="wooden_two_layer_shelf",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/mjcf/wooden_two_layer_shelf.xml",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/urdf/wooden_two_layer_shelf.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_two_layer_shelf/mjcf/wooden_two_layer_shelf.xml",
            ),
            ArticulationObjCfg(
                name="flat_stove",
                fix_base_link=True,
                usd_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/usd/flat_stove.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/urdf/flat_stove.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/articulated_objects/flat_stove/mjcf/flat_stove.xml",
            ),
        ],
        robots=["franka"],
        # Scene configuration,
    )

    # Task parameters
    max_episode_steps = 400
    task_desc = "Put the white bowl on top of the cabinet (scene9)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.05),)

    # Trajectory file path
    traj_filepath = "roboverse_data/trajs/libero90/libero_90_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_traj_v2.pkl"

    goal = (
        ("Turnon", "flat_stove", "button"),
        ("On", "chefmate_8_frypan", "flat_stove", "cook_region"),
    )
