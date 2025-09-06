from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class StackWineCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_wine/v2"
    objects = [
        RigidObjCfg(
            name="wine_bottle_visual",
            usd_path="roboverse_data/assets/rlbench/stack_wine/wine_bottle_visual/usd/wine_bottle_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="rack_bottom_visual",
            usd_path="roboverse_data/assets/rlbench/stack_wine/rack_bottom_visual/usd/rack_bottom_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="rack_top_visual",
            usd_path="roboverse_data/assets/rlbench/stack_wine/rack_top_visual/usd/rack_top_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
