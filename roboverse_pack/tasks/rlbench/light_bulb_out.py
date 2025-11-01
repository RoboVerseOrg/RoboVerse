from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.light_bulb_out", "light_bulb_out", "franka.light_bulb_out")
class LightBulbOutTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bulb",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb0/usd/bulb0.usd",  # reuse light_bulb_in asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bulb_holder0",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder0/usd/bulb_holder0.usd",  # reuse light_bulb_in asset
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="bulb_holder1",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/bulb_holder1/usd/bulb_holder1.usd",  # reuse light_bulb_in asset
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="lamp_base",
                usd_path="roboverse_data/assets/rlbench/light_bulb_in/lamp_base/usd/lamp_base.usd",  # reuse light_bulb_in asset
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/light_bulb_out/v2/franka_v2.pkl.gz"
    # TODO: add checker
