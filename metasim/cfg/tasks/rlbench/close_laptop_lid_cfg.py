from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class CloseLaptopLidCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_laptop_lid/v2"
    objects = [
        ArticulationObjCfg(
            name="base",
            usd_path="roboverse_data/assets/rlbench/close_laptop_lid/base/usd/base.usd",
        ),
        RigidObjCfg(
            name="laptop_holder",
            usd_path="roboverse_data/assets/rlbench/close_laptop_lid/laptop_holder/usd/laptop_holder.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
