from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class HangFrameOnHangerTask(RLBenchTask):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/hang_frame_on_hangerv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="frame",
            usd_path="roboverse_data/assets/rlbench/hang_frame_on_hanger/frame/usd/frame.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="hanger",
            usd_path="roboverse_data/assets/rlbench/hang_frame_on_hanger/hanger/usd/hanger.usd",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="task_wall",
            usd_path="roboverse_data/assets/rlbench/hang_frame_on_hanger/task_wall/usd/task_wall.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
