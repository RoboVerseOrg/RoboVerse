from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class StackChairsTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_chairsv2/franka_v2.pkl.gz"
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
