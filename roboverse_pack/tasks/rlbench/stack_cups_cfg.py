from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class StackCupsTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_cupsv2/franka_v2.pkl.gz"
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
