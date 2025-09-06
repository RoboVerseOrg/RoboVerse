from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

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
class PlaceCupsCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/place_cups/v2"
    objects = [_CUP_HOLDER] + _CUPS
    # TODO: add checker


@configclass
class RemoveCupsCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/remove_cups/v2"
    objects = [_CUP_HOLDER] + _CUPS
    # TODO: add checker
