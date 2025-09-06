from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutShoesInBoxTask(RLBenchTask):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/put_shoes_in_boxv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="box_base",
            usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/box_base/usd/box_base.usd",
        ),
        RigidObjCfg(
            name="shoe1_visual",
            usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/shoe1_visual/usd/shoe1_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="shoe2_visual",
            usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/shoe2_visual/usd/shoe2_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
