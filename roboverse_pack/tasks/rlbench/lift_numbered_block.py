from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.lift_numbered_block", "lift_numbered_block", "franka.lift_numbered_block")
class LiftNumberedBlockTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="block1",
                usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block1/usd/block1.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="block2",
                usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block2/usd/block2.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="block3",
                usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block3/usd/block3.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/lift_numbered_block/v2/franka_v2.pkl.gz"
    # TODO: add checker
