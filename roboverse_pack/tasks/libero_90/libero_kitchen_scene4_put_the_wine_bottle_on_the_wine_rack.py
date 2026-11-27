"""Configuration for the Libero kitchen scene4 put the wine bottle on the wine rack task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.kitchen_scene4_put_the_wine_bottle_on_the_wine_rack",
    "kitchen_scene4_put_the_wine_bottle_on_the_wine_rack",
)
class LiberoKitchenScene4PutWineBottleOnWineRackTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene4 put the wine bottle on the wine rack task.

    This task is transferred from:
    KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl

    Task Description:
    - Put the wine bottle on the wine rack

    This is a manipulation task that requires:
    1. Grasping wine_bottle
    2. Placing it on the wine rack

    Objects from BDDL:
    - white_cabinet (fixture): The cabinet with drawers
    - wine_bottle(object): Bottle to be placed on the rack
    - akita_black_bowl(object): Distractor object
    - wine_rack(object): Target rack

    Goal: (On wine_bottle wine_rack_top_surface)
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
    )

    max_episode_steps = 300
    task_desc = "Put the wine bottle on the wine rack (scene4)"

    workspace_name = ("kitchen_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.05),)

    traj_filepath = (
        "roboverse_data/trajs/libero90/libero_90_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_traj_v2.pkl"
    )

    goal = (("On", "wine_bottle", "wine_rack", "top_region"),)
