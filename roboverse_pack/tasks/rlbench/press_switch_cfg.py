from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PressSwitchTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/press_switchv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="switch_main",
            usd_path="roboverse_data/assets/rlbench/press_switch/switch_main/usd/switch_main.usd",
        ),
        RigidObjCfg(
            name="task_wall",
            usd_path="roboverse_data/assets/rlbench/press_switch/task_wall/usd/task_wall.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
