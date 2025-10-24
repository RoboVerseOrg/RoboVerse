from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_money_in_safe", "put_money_in_safe", "franka.put_money_in_safe")
class PutMoneyInSafeTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="dollar_stack",
                usd_path="roboverse_data/assets/rlbench/put_money_in_safe/dollar_stack/usd/dollar_stack.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="safe_body",
                usd_path="roboverse_data/assets/rlbench/put_money_in_safe/safe_body/usd/safe_body.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_money_in_safe/v2/franka_v2.pkl.gz"
    # TODO: add checker
