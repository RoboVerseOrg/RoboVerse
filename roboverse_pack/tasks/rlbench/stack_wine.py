from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.stack_wine", "stack_wine", "franka.stack_wine")
class StackWineTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
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
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/stack_wine/v2/franka_v2.pkl.gz"
    # TODO: add checker
