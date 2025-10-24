from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.beat_the_buzz", "beat_the_buzz", "franka.beat_the_buzz")
class BeatTheBuzzTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="wand",
                usd_path="roboverse_data/assets/rlbench/beat_the_buzz/wand/usd/wand.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="Cuboid",
                usd_path="roboverse_data/assets/rlbench/beat_the_buzz/Cuboid/usd/Cuboid.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/beat_the_buzz/v2/franka_v2.pkl.gz"
    # TODO: add checker
