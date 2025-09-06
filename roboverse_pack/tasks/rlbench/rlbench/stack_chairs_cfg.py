from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class StackChairsCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_chairs/v2"
    objects = [
        RigidObjCfg(
            name="chair1",
            usd_path="roboverse_data/assets/rlbench/stack_chairs/chair1/usd/chair1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="chair2",
            usd_path="roboverse_data/assets/rlbench/stack_chairs/chair2/usd/chair2.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="chair3",
            usd_path="roboverse_data/assets/rlbench/stack_chairs/chair3/usd/chair3.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
