from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.set_the_table", "set_the_table", "franka.set_the_table")
class SetTheTableTask(RLBenchTask):
    max_episode_steps = 1200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="holder",
                usd_path="roboverse_data/assets/rlbench/set_the_table/holder/usd/holder.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="fork_visual",
                usd_path="roboverse_data/assets/rlbench/set_the_table/fork_visual/usd/fork_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="knife_visual",
                usd_path="roboverse_data/assets/rlbench/set_the_table/knife_visual/usd/knife_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="spoon_visual",
                usd_path="roboverse_data/assets/rlbench/set_the_table/spoon_visual/usd/spoon_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="plate_visual",
                usd_path="roboverse_data/assets/rlbench/set_the_table/plate_visual/usd/plate_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="glass_visual",
                usd_path="roboverse_data/assets/rlbench/set_the_table/glass_visual/usd/glass_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/set_the_table/v2/franka_v2.pkl.gz"
    # TODO: add checker
