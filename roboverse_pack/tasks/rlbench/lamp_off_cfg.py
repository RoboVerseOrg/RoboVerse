from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

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
class LampOffTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_offv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class LampOnTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_onv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
