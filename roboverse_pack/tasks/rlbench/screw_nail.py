from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.screw_nail", "screw_nail", "franka.screw_nail")
class ScrewNailTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="block",
                usd_path="roboverse_data/assets/rlbench/screw_nail/block/usd/block.usd",
            ),
            RigidObjCfg(
                name="screw_driver",
                usd_path="roboverse_data/assets/rlbench/screw_nail/screw_driver/usd/screw_driver.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/screw_nail/v2/franka_v2.pkl.gz"
    # TODO: add checker
