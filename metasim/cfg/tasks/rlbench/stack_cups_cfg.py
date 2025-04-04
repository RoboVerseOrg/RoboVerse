from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class StackCupsCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_cups/v2"
    objects = [
        RigidObjCfg(
            name="cup1_visual",
            usd_path="roboverse_data/assets/rlbench/stack_cups/cup1_visual/usd/cup1_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup2_visual",
            usd_path="roboverse_data/assets/rlbench/stack_cups/cup2_visual/usd/cup2_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="cup3_visual",
            usd_path="roboverse_data/assets/rlbench/stack_cups/cup3_visual/usd/cup3_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
