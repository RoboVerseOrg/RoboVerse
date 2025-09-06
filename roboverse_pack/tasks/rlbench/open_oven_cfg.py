from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

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
class OpenOvenTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_ovenv2/franka_v2.pkl.gz"
    objects = [_OVEN]
    # TODO: add checker


@configclass
class PutTrayInOvenTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_tray_in_ovenv2/franka_v2.pkl.gz"
    objects = [_OVEN, _TRAY]
    # TODO: add checker


@configclass
class TakeTrayOutOfOvenTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/take_tray_out_of_ovenv2/franka_v2.pkl.gz"
    objects = [_OVEN, _TRAY]
    # TODO: add checker
