from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.light_bulb_in", "light_bulb_in", "franka.light_bulb_in")
class LightBulbInTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bulb0",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bulb1",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb1/usd/bulb1.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bulb_holder0",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="bulb_holder1",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="lamp_base",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/light_bulb_in/v2/franka_v2.pkl.gz"
    # TODO: add checker
