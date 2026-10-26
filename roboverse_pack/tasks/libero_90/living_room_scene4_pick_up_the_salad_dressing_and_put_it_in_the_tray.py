"""Configuration for the Libero living room scene4 pick up the salad dressing and put it in the tray task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    "living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
)
class LiberoLivingRoomScene4PickUpSaladDressingTask(Libero90BaseTask):
    """Configuration for the Libero living room scene4 pick up the salad dressing and put it in the tray task.

    Task Description:
    - Pick up the new_salad_dressing from the table
    - Place the new_salad_dressing inside the wooden_tray/contain_region

    This is a manipulation task that requires:
    1. Picking up the new_salad_dressing from the table
    2. Placing the new_salad_dressing inside the wooden_tray/contain_region

    Objects:
    - new_salad_dressing (target)
    - akita_black_bowl_1 (left bowl)
    - akita_black_bowl_2 (right bowl)
    - chocolate_pudding
    - wooden_tray (goal container)

    Goal: Place new_salad_dressing inside wooden_tray/contain_region.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="new_salad_dressing",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/usd/new_salad_dressing.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/urdf/new_salad_dressing.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/new_salad_dressing/mjcf/new_salad_dressing.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl_1",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="akita_black_bowl_2",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/usd/akita_black_bowl.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/urdf/akita_black_bowl.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/akita_black_bowl/mjcf/akita_black_bowl.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chocolate_pudding",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/usd/chocolate_pudding.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/urdf/chocolate_pudding.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/mjcf/chocolate_pudding.xml",
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
    task_desc = "Pick up the salad_dressing and put it in the tray (living_room_scene4)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0.0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_traj_v2.pkl"

    goal = (("In", "new_salad_dressing", "wooden_tray", "contain_region"),)
