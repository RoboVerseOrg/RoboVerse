"""The plug charger task from ManiSkill."""

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers import DetectedChecker, RelativeBboxDetector
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.plug_charger", "plug_charger")
class PlugChargerTask(ManiskillBaseTask):
    """The plug charger task from ManiSkill.

    The robot is tasked to plug the charger into the base.
    The checker is to check if the charger is plugged into the base.
    Note that the two holes on the socket asset are slightly enlarged to reduce the difficulty due to the dynamics gap.
    """

    max_episode_steps = 250
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="base",
                usd_path="roboverse_data/assets/maniskill/charger/base/base.usd",
                urdf_path="roboverse_data/assets/maniskill/charger/base.urdf",
                physics=PhysicStateType.GEOM,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="charger",
                usd_path="roboverse_data/assets/maniskill/charger/charger/charger.usd",
                urdf_path="roboverse_data/assets/maniskill/charger/charger.urdf",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ]
    )
    traj_filepath = "roboverse_data/trajs/maniskill/plug_charger/trajectory-franka_v2.pkl"

    checker = DetectedChecker(
        obj_name="charger",
        detector=RelativeBboxDetector(
            base_obj_name="base",
            relative_quat=[1, 0, 0, 0],
            relative_pos=[0, 0, 0],
            checker_lower=[-0.02, -0.075, -0.075],
            checker_upper=[0.02, 0.075, 0.075],
            # debug_vis=True,
        ),
    )
