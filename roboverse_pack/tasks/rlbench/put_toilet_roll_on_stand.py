from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_toilet_roll_on_stand", "put_toilet_roll_on_stand", "franka.put_toilet_roll_on_stand")
class PutToiletRollOnStandTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="toilet_roll_visual",
                usd_path="roboverse_data/assets/rlbench/put_toilet_roll_on_stand/toilet_roll_visual/usd/toilet_roll_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="stand_base",
                usd_path="roboverse_data/assets/rlbench/put_toilet_roll_on_stand/stand_base/usd/stand_base.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveCubeCfg(
                name="toilet_roll_box",
                size=[0.1, 0.1, 0.1],
                color=[0.85, 0.85, 0.85],
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_toilet_roll_on_stand/v2/franka_v2.pkl.gz"
    # TODO: add checker
