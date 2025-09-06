from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

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
class MeatOffGrillTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/meat_off_grillv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class MeatOnGrillTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/meat_on_grillv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
