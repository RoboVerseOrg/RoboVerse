"""Configuration for the Libero pick bbq sauce task."""

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .libero_base import LiberoBaseTask


@register_task("libero.pick_bbq_sauce", "pick_bbq_sauce")
class LiberoPickBbqSauceTask(LiberoBaseTask):
    """Configuration for the Libero pick bbq sauce task.

    This task is transferred from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl
    """

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bbq_sauce",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            RigidObjCfg(
                name="basket",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/usd/basket.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/urdf/basket.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/basket/mjcf/basket.xml",
            ),
            RigidObjCfg(
                name="chocolate_pudding",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/usd/chocolate_pudding.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/urdf/chocolate_pudding.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/chocolate_pudding/mjcf/chocolate_pudding.xml",
            ),
            RigidObjCfg(
                name="ketchup",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/urdf/ketchup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/mjcf/ketchup.xml",
            ),
            RigidObjCfg(
                name="salad_dressing",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/salad_dressing/usd/salad_dressing.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/salad_dressing/urdf/salad_dressing.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/salad_dressing/mjcf/salad_dressing.xml",
            ),
            RigidObjCfg(
                name="alphabet_soup",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/urdf/alphabet_soup.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/alphabet_soup/mjcf/alphabet_soup.xml",
            ),
            RigidObjCfg(
                name="cream_cheese",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/usd/cream_cheese.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/urdf/cream_cheese.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/cream_cheese/mjcf/cream_cheese.xml",
            ),
        ]
    )

    # task horizon
    max_episode_steps = 250
    task_desc = "Pick up the bbq sauce and place it in the basket"
    checker = DetectedChecker(
        obj_name="bbq_sauce",
        detector=RelativeBboxDetector(
            base_obj_name="basket",
            relative_pos=[0.0, 0.0, 0.07185],
            relative_quat=[1.0, 0.0, 0.0, 0.0],
            checker_lower=[-0.08, -0.08, -0.11],
            checker_upper=[0.08, 0.08, 0.05],
        ),
    )
    traj_filepath = "roboverse_data/trajs/libero/pick_up_the_bbq_sauce_and_place_it_in_the_basket/v2/franka_v2.pkl.gz"
