from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.reach_and_drag", "reach_and_drag", "franka.reach_and_drag")
class ReachAndDragTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.08, 0.08, 0.08],
                color=[0.85, 0.85, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveCubeCfg(
                name="stick",
                size=[0.0128, 0.0128, 0.36],
                color=[0.85, 0.85, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="target0",
                usd_path="roboverse_data/assets/rlbench/reach_and_drag/target0/usd/target0.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/reach_and_drag/v2/franka_v2.pkl.gz"
    # TODO: add checker
