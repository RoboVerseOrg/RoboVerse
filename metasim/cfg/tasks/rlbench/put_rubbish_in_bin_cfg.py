from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PutRubbishInBinCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/put_rubbish_in_bin/v2"
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
