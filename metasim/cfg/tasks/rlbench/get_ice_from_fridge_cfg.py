from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class GetIceFromFridgeCfg(RLBenchTaskCfg):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/get_ice_from_fridge/v2"
    objects = [
        ArticulationObjCfg(
            name="fridge_base",
            usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/fridge_base/usd/fridge_base.usd",
        ),
        RigidObjCfg(
            name="cup_visual",
            usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/cup_visual/usd/cup_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
