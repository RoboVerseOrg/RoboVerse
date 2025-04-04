from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_OVEN = ArticulationObjCfg(
    name="oven_base",
    usd_path="roboverse_data/assets/rlbench/open_oven/oven_base/usd/oven_base.usd",
)
_TRAY = RigidObjCfg(
    name="tray_visual",
    usd_path="roboverse_data/assets/rlbench/put_tray_in_oven/tray_visual/usd/tray_visual.usd",
    physics=PhysicStateType.RIGIDBODY,
)


@configclass
class OpenOvenCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_oven/v2"
    objects = [_OVEN]
    # TODO: add checker


@configclass
class PutTrayInOvenCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_tray_in_oven/v2"
    objects = [_OVEN, _TRAY]
    # TODO: add checker


@configclass
class TakeTrayOutOfOvenCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/take_tray_out_of_oven/v2"
    objects = [_OVEN, _TRAY]
    # TODO: add checker
