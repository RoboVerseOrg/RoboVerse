from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.stack_chairs", "stack_chairs", "franka.stack_chairs")
class StackChairsTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="chair1",
                usd_path="roboverse_data/assets/rlbench/stack_chairs/chair1/usd/chair1.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chair2",
                usd_path="roboverse_data/assets/rlbench/stack_chairs/chair2/usd/chair2.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="chair3",
                usd_path="roboverse_data/assets/rlbench/stack_chairs/chair3/usd/chair3.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/stack_chairs/v2/franka_v2.pkl.gz"
    # TODO: add checker
