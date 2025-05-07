"""The pick up cube task from ManiSkill."""

from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .maniskill_task_cfg import ManiskillTaskCfg


@configclass
class PickCubeCfg(ManiskillTaskCfg):
    """The pick up cube task from ManiSkill.

    The robot is tasked to pick up a cube.
    Note that the checker is not same as the original one (checking if the cube is near the target position).
    The current one checks if the cube is lifted up 0.1 meters.
    """

    episode_length = 250
    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=[0.04, 0.04, 0.04],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[1.0, 0.0, 0.0],
        ),
    ]
    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/trajectory-unified-retarget_v2.pkl"
    checker = PositionShiftChecker(
        obj_name="cube",
        distance=0.1,
        axis="z",
    )
