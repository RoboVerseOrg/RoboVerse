from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PutRubbishInBinTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_rubbish_in_binv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="bin_visual",
            usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/bin_visual/usd/bin_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tomato1_visual",
            usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/tomato1_visual/usd/tomato1_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="tomato2_visual",
            usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/tomato2_visual/usd/tomato2_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="rubbish_visual",
            usd_path="roboverse_data/assets/rlbench/put_rubbish_in_bin/rubbish_visual/usd/rubbish_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
