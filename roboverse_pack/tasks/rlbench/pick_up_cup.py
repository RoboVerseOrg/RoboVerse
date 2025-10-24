from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.pick_up_cup", "pick_up_cup", "franka.pick_up_cup")
class PickUpCupTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cup1_visual",
                usd_path="roboverse_data/assets/rlbench/pick_up_cup/cup1_visual/usd/cup1_visual.usd",
                physics=PhysicStateType.XFORM,
            ),
            RigidObjCfg(
                name="cup2_visual",
                usd_path="roboverse_data/assets/rlbench/pick_up_cup/cup2_visual/usd/cup2_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/pick_up_cup/v2/franka_v2.pkl.gz"
    # TODO: add checker
