"""The plug charger task from ManiSkill."""

from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class PlugChargerCfg(ManiskillTaskCfg):
    """The plug charger task from ManiSkill.

    The robot is tasked to plug the charger into the base.
    The checker is to check if the charger is plugged into the base.
    Note that the two holes on the socket asset are slightly enlarged to reduce the difficulty due to the dynamics gap.
    """

    episode_length = 250
    objects = [
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
