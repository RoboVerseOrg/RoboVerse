from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_OBJECTS = [
    RigidObjCfg(
        name="grill_visual",
        usd_path="roboverse_data/assets/rlbench/meat_off_grill/grill_visual/usd/grill_visual.usd",
        physics=PhysicStateType.GEOM,
    ),
    RigidObjCfg(
        name="chicken_visual",
        usd_path="roboverse_data/assets/rlbench/meat_off_grill/chicken_visual/usd/chicken_visual.usd",
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="steak_visual",
        usd_path="roboverse_data/assets/rlbench/meat_off_grill/steak_visual/usd/steak_visual.usd",
        physics=PhysicStateType.RIGIDBODY,
    ),
]


@configclass
class MeatOffGrillCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/meat_off_grill/v2"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class MeatOnGrillCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/meat_on_grill/v2"
    objects = _OBJECTS
    # TODO: add checker
