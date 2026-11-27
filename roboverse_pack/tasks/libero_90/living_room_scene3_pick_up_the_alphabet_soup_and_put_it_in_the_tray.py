"""Configuration for the Libero living room scene3 pick up the alphabet soup and put it in the tray task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
    "living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
)
class LiberoLivingRoomScene3PickUpAlphabetSoupTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene10 put the chocolate pudding in the top drawer of the cabinet and close it task.

    Task Description:
    - Pick up the alphabet_soup from the table
    - Place the alphabet_soup inside the tray/contain_region

    This is a compound manipulation task that requires:
    1. Picking up the alphabet_soup from the table
    2. Placing the alphabet_soup inside the tray/contain_region

    Objects:
    - alphabet_soup (target)
    - cream_cheese
    - tomato_sauce
    - ketchup
    - butter
    - wooden_tray (goal container)

    Goal: Place alphabet_soup inside wooden_tray.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="alphabet_soup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/urdf/alphabet_soup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/mjcf/alphabet_soup.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="cream_cheese",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/urdf/cream_cheese.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/mjcf/cream_cheese.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="tomato_sauce",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/urdf/tomato_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/mjcf/tomato_sauce.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="ketchup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="butter",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="wooden_tray",
                usd_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/usd/wooden_tray.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/urdf/wooden_tray.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/turbosquid_objects/wooden_tray/mjcf/wooden_tray.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 200
    task_desc = "Pick up the alphabet_soup and put it in the tray (living_room_scene3)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_traj_v2.pkl"

    goal = (("In", "alphabet_soup", "wooden_tray", "contain_region"),)
