from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.scoop_with_spatula", "scoop_with_spatula", "franka.scoop_with_spatula")
class ScoopWithSpatulaTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="spatula_visual",
                usd_path="roboverse_data/assets/rlbench/scoop_with_spatula/spatula_visual/usd/spatula_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveCubeCfg(
                name="Cuboid",
                size=[0.02, 0.02, 0.02],
                color=[0.85, 0.85, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/scoop_with_spatula/v2/franka_v2.pkl.gz"
    # TODO: add checker
