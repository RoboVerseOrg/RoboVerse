"""Configuration for the Libero kitchen scene8 put the right moka pot on the stove task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene8_put_the_right_moka_pot_on_the_stove",
    "kitchen_scene8_put_the_right_moka_pot_on_the_stove",
)
class LiberoKitchenScene8PutTheRightMokaPotOnTheStoveTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene8 put the right moka pot on the stove task.

    Task Description:
    - Put the right moka pot (moka_pot_2) on the stove

    This is a manipulation task that requires:
    1. Identifying and grasping moka_pot_2 (the right one)
    2. Placing it on top of the flat_stove

    Objects:
    - moka_pot_1 (object): Distractor moka pot (left one)
    - moka_pot_2 (object): Target moka pot (right one) to be placed on the stove
    - flat_stove (fixture): Target stove surface

    Goal: (On moka_pot_2 flat_stove)
    """

    scenario = ScenarioCfg(
        objects=[
            # Movable objects - two moka pots
            RigidObjCfg(
                name="moka_pot_1",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/usd/moka_pot.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/urdf/moka_pot.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="moka_pot_2",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/usd/moka_pot.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/urdf/moka_pot.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/moka_pot/mjcf/moka_pot.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            # Fixed fixtures
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
    max_episode_steps = 300  # Moderate complexity task
    task_desc = "Put the right moka pot on the stove (scene8)"

    # Workspace configuration
    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)  # kitchen_table_offset
    workspace_size = ((1.0, 1.2, 0.05),)  # kitchen_table_full_size

    # Trajectory file path
    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene8_put_the_right_moka_pot_on_the_stove_traj_v2.pkl"
    )

    goal = (
        ("On", "moka_pot_1", "flat_stove", "cook_region"),
        ("Turnon", "flat_stove", "button"),
    )
