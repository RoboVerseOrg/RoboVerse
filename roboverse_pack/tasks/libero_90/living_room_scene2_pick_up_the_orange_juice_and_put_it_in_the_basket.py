"""Configuration for the Libero living room scene2 pick up the orange juice and put it in the basket task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket",
    "living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket",
)
class LiberoLivingRoomScene2PickUpOrangeJuiceTask(Libero90BaseTask):
    """Configuration for the Libero kitchen scene10 put the chocolate pudding in the top drawer of the cabinet and close it task.

    Task Description:
    - Pick up the orange_juice from the table
    - Place the orange_juice inside the basket/contain_region

    This is a compound manipulation task that requires:
    1. Picking up the orange_juice from the table
    2. Placing the orange_juice inside the basket/contain_region

    Objects:
    - alphabet_soup
    - cream_cheese
    - tomato_sauce
    - ketchup
    - orange_juice (target)
    - milk
    - butter
    - basket (goal container)

    Goal: Place orange_juice inside basket/contain_region.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="orange_juice",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/usd/orange_juice.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/urdf/orange_juice.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/mjcf/orange_juice.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
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
                name="milk",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/usd/milk.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/urdf/milk.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/milk/mjcf/milk.xml",
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
                name="basket",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/mjcf/basket.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 300
    task_desc = "Pick up the orange_juice and put it in the basket (living_room_scene2)"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket_traj_v2.pkl"

    goal = (("In", "orange_juice", "basket", "contain_region"),)
