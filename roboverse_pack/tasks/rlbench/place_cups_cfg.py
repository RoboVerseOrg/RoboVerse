from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

_CUP_HOLDER = RigidObjCfg(
    name="place_cups_holder_base",
    usd_path="roboverse_data/assets/rlbench/place_cups/place_cups_holder_base/usd/place_cups_holder_base.usd",
    physics=PhysicStateType.XFORM,
)

_CUPS = [
    RigidObjCfg(
        name=f"mug_visual{i}",
        usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
        physics=PhysicStateType.RIGIDBODY,
    )
    for i in range(4)
]


@configclass
class PlaceCupsTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/place_cupsv2/franka_v2.pkl.gz"
    objects = [_CUP_HOLDER] + _CUPS
    # TODO: add checker


@configclass
class RemoveCupsTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/remove_cupsv2/franka_v2.pkl.gz"
    objects = [_CUP_HOLDER] + _CUPS
    # TODO: add checker
