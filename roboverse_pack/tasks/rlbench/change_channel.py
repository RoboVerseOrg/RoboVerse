from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.change_channel", "change_channel", "franka.change_channel")
class ChangeChannelTask(RLBenchTask):
    max_episode_steps = 200
    traj_filepath = "roboverse_data/trajs/rlbench/change_channel/v2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="tv_remote",
            usd_path="roboverse_data/assets/rlbench/change_channel/tv_remote/usd/tv_remote.usd",
        ),
        RigidObjCfg(
            name="tv_frame",
            usd_path="roboverse_data/assets/rlbench/change_channel/tv_frame/usd/tv_frame.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
