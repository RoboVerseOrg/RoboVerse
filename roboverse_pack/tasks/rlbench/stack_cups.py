from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.stack_cups", "stack_cups", "franka.stack_cups")
class StackCupsTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cup1_visual",
                usd_path="roboverse_data/assets/rlbench/stack_cups/cup1_visual/usd/cup1_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="cup2_visual",
                usd_path="roboverse_data/assets/rlbench/stack_cups/cup2_visual/usd/cup2_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="cup3_visual",
                usd_path="roboverse_data/assets/rlbench/stack_cups/cup3_visual/usd/cup3_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/stack_cups/v2/franka_v2.pkl.gz"
    # TODO: add checker
