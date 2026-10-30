"""Configuration for the Libero living room scene1 pick up the tomato sauce and put it in the basket task."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_90_base import Libero90BaseTask


@register_task(
    "libero_90.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
)
class LiberoLivingRoomScene1PickUpTomatoSauceAndPutItInTheBasketTask(Libero90BaseTask):
    """Configuration for the Libero living room scene1 pick up the tomato sauce and put it in the basket task.

    Objects:
    - tomato_sauce (target)
    - basket (goal container)

    Goal: Place tomato_sauce inside basket/contain_region.
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="ketchup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
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
                name="alphabet_soup",
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/urdf/alphabet_soup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/mjcf/alphabet_soup.xml",
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
                name="basket",
                usd_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_scanned_objects/basket/mjcf/basket.xml",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )

    max_episode_steps = 250
    task_desc = "Pick up the tomato_sauce and put it in the basket"

    workspace_name = ("living_room_table",)
    workspace_offset = ((0, 0, 0),)
    workspace_size = ((1.0, 1.2, 0.1),)

    traj_filepath = "roboverse_data/trajs/libero90/libero_90_living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_traj_v2.pkl"

    goal = (("In", "tomato_sauce", "basket", "contain_region"),)
