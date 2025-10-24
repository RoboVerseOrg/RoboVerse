from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.slide_block_to_target", "slide_block_to_target", "franka.slide_block_to_target")
class SlideBlockToTargetTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="block",
                size=[0.05625, 0.05625, 0.05625],
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="target",
                usd_path="roboverse_data/assets/rlbench/slide_block_to_target/target/usd/target.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/slide_block_to_target/v2/franka_v2.pkl.gz"
    # TODO: add checker
