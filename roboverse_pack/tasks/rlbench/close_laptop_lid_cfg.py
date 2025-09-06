from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class CloseLaptopLidTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_laptop_lidv2/franka_v2.pkl.gz"
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
