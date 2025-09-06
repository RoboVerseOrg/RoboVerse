from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_CABINET = ArticulationObjCfg(
    name="cabinet_base",
    usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cabinet_base/usd/cabinet_base.usd",
)

_CUP = RigidObjCfg(
    name="cup_visual",
    usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cup_visual/usd/cup_visual.usd",
    physics=PhysicStateType.RIGIDBODY,
)


@configclass
class SlideCabinetOpenAndPlaceCupsCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/slide_cabinet_open_and_place_cups/v2"
    objects = [_CABINET, _CUP]
    # TODO: add checker


@configclass
class TakeCupOutFromCabinetCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/take_cup_out_from_cabinet/v2"
    objects = [_CABINET, _CUP]
    # TODO: add checker
