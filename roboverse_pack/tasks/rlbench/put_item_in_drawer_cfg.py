from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutItemInDrawerTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_item_in_drawerv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="drawer_frame",
            usd_path="roboverse_data/assets/rlbench/put_item_in_drawer/drawer_frame/usd/drawer_frame.usd",
        ),
        PrimitiveCubeCfg(
            name="item",
            size=[0.04, 0.04, 0.04],
            color=[0.85, 0.85, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
