from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.reach_target", "reach_target", "franka.reach_target")
class ReachTargetTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            PrimitiveSphereCfg(
                name="target",
                radius=0.025,
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.XFORM,
            ),
            PrimitiveSphereCfg(
                name="distractor0",
                radius=0.025,
                color=[1.0, 0.0, 0.5],
                physics=PhysicStateType.XFORM,
            ),
            PrimitiveSphereCfg(
                name="distractor1",
                radius=0.025,
                color=[1.0, 1.0, 0.0],
                physics=PhysicStateType.XFORM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/reach_target/v2/franka_v2.pkl.gz"
    # TODO: add checker
