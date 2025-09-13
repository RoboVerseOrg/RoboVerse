from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_base import LiberoBaseTask


@register_task("libero.pick_butter", "pick_butter")
class LiberoPickButterTask(LiberoBaseTask):
    """Configuration for the Libero pick butter task.

    This task is transferred from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_object/pick_up_the_butter_and_place_it_in_the_basket.bddl
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="butter",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/urdf/butter.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/mjcf/butter.xml",
            ),
            RigidObjCfg(
                name="basket",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/mjcf/basket.xml",
            ),
            RigidObjCfg(
                name="tomato_sauce",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/usd/tomato_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/urdf/tomato_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/tomato_sauce/mjcf/tomato_sauce.xml",
            ),
            RigidObjCfg(
                name="orange_juice",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/usd/orange_juice.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/urdf/orange_juice.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/mjcf/orange_juice.xml",
            ),
            RigidObjCfg(
                name="chocolate_pudding",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/usd/chocolate_pudding.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/urdf/chocolate_pudding.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/mjcf/chocolate_pudding.xml",
            ),
            RigidObjCfg(
                name="bbq_sauce",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            RigidObjCfg(
                name="ketchup",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
            ),
        ],
        robots=["franka"],
    )

    # task horizon
    max_episode_steps = 250
    task_desc = "Pick up the butter and place it in the basket"
    checker = DetectedChecker(
        obj_name="butter",
        detector=RelativeBboxDetector(
            base_obj_name="basket",
            relative_pos=[0.0, 0.0, 0.07185],
            relative_quat=[1.0, 0.0, 0.0, 0.0],
            checker_lower=[-0.08, -0.08, -0.11],
            checker_upper=[0.08, 0.08, 0.05],
        ),
    )
    traj_filepath = "roboverse_data/trajs/libero/pick_up_the_butter_and_place_it_in_the_basket/v2/franka_v2.pkl.gz"
