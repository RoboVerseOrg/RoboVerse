from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class CloseJarTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_jarv2/franka_v2.pkl.gz"
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
