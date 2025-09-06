from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class CloseJarCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_jar/v2"
    objects = [
        RigidObjCfg(
            name="jar0",
            usd_path="roboverse_data/assets/rlbench/close_jar/jar0/usd/jar0.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="jar1",
            usd_path="roboverse_data/assets/rlbench/close_jar/jar1/usd/jar1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="jar_lid0",
            usd_path="roboverse_data/assets/rlbench/close_jar/jar_lid0/usd/jar_lid0.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
