from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_OBJECTS = [
    RigidObjCfg(
        name="lamp_base",
        usd_path="roboverse_data/assets/rlbench/lamp_off/lamp_base/usd/lamp_base.usd",
        physics=PhysicStateType.GEOM,
    ),
    ArticulationObjCfg(
        name="push_button_target",
        usd_path="roboverse_data/assets/rlbench/lamp_off/push_button_target/usd/push_button_target.usd",
    ),
]


@configclass
class LampOffCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_off/v2"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class LampOnCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_on/v2"
    objects = _OBJECTS
    # TODO: add checker
