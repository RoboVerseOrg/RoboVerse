from metasim.cfg.objects import PrimitiveCylinderCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg
from .setup_chess_cfg import _CHESS_BOARD

_WHITE_CHECKERS = [
    PrimitiveCylinderCfg(
        name=f"checker{idx}",
        radius=0.017945,
        height=0.00718,
        color=[1.0, 1.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    )
    for idx in range(12)
]

_RED_CHECKERS = [
    PrimitiveCylinderCfg(
        name=f"checker{idx}",
        radius=0.017945,
        height=0.00718,
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    )
    for idx in range(12, 24)
]


@configclass
class SetupCheckersCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/setup_checkers/v2"
    objects = [_CHESS_BOARD] + _WHITE_CHECKERS + _RED_CHECKERS
    # TODO: add checker
