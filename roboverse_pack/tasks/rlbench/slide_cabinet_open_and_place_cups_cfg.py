from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

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
class SlideCabinetOpenAndPlaceCupsTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/slide_cabinet_open_and_place_cupsv2/franka_v2.pkl.gz"
    objects = [_CABINET, _CUP]
    # TODO: add checker


@configclass
class TakeCupOutFromCabinetTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/take_cup_out_from_cabinetv2/franka_v2.pkl.gz"
    objects = [_CABINET, _CUP]
    # TODO: add checker
