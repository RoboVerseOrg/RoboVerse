from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutToiletRollOnStandTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_toilet_roll_on_standv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="toilet_roll_visual",
            usd_path="roboverse_data/assets/rlbench/put_toilet_roll_on_stand/toilet_roll_visual/usd/toilet_roll_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="stand_base",
            usd_path="roboverse_data/assets/rlbench/put_toilet_roll_on_stand/stand_base/usd/stand_base.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="toilet_roll_box",
            size=[0.1, 0.1, 0.1],
            color=[0.85, 0.85, 0.85],
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
