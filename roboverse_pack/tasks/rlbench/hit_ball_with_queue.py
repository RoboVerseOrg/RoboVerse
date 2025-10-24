from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.hit_ball_with_queue", "hit_ball_with_queue", "franka.hit_ball_with_queue")
class HitBallWithQueueTask(RLBenchTask):
    max_episode_steps = 600
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="queue",
                usd_path="roboverse_data/assets/rlbench/hit_ball_with_queue/queue/usd/queue.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="ball",
                physics=PhysicStateType.RIGIDBODY,
                color=[0.96, 0.96, 0.96],
                radius=0.01,
            ),
            PrimitiveCubeCfg(
                name="hit_ball_with_queue_pocket",
                size=[0.0075, 0.08, 0.005],
                physics=PhysicStateType.GEOM,
                color=[0.54, 1.0, 0.51],
            ),
            PrimitiveCubeCfg(
                name="hit_ball_with_queue_pocket0",
                size=[0.0075, 0.08, 0.005],
                physics=PhysicStateType.GEOM,
                color=[0.54, 1.0, 0.51],
            ),
            PrimitiveCubeCfg(
                name="hit_ball_with_queue_pocket1",
                size=[0.0075, 0.04, 0.005],
                physics=PhysicStateType.GEOM,
                color=[0.54, 1.0, 0.51],
            ),
            PrimitiveCubeCfg(
                name="hit_ball_with_queue_stopper1",
                size=[0.0004, 0.016, 0.0004],
                physics=PhysicStateType.GEOM,
                color=[0.85, 0.85, 1.00],
            ),
            PrimitiveCubeCfg(
                name="hit_ball_with_queue_stopper2",
                size=[0.0004, 0.016, 0.0004],
                physics=PhysicStateType.GEOM,
                color=[0.85, 0.85, 1.00],
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/hit_ball_with_queue/v2/franka_v2.pkl.gz"
    # TODO: add checker
