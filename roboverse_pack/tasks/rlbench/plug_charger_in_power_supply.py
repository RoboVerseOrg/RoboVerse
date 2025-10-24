from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task(
    "rlbench.plug_charger_in_power_supply", "plug_charger_in_power_supply", "franka.plug_charger_in_power_supply"
)
class PlugChargerInPowerSupplyTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="charger",
                usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/charger/usd/charger.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="task_wall",
                usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/task_wall/usd/task_wall.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="plug",
                usd_path="roboverse_data/assets/rlbench/plug_charger_in_power_supply/plug/usd/plug.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/plug_charger_in_power_supply/v2/franka_v2.pkl.gz"
    # TODO: add checker
