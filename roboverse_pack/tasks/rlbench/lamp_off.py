from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.lamp_off", "lamp_off", "franka.lamp_off")
class LampOffTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="lamp_base",
                usd_path="roboverse_data/assets/rlbench/lamp_off/lamp_base/usd/lamp_base.usd",
                physics=PhysicStateType.GEOM,
            ),
            ArticulationObjCfg(
                name="push_button_target",
                usd_path="roboverse_data/assets/rlbench/lamp_off/push_button_target/usd/push_button_target.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_off/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.lamp_on", "lamp_on", "franka.lamp_on")
class LampOnTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="lamp_base",
                usd_path="roboverse_data/assets/rlbench/lamp_off/lamp_base/usd/lamp_base.usd",
                physics=PhysicStateType.GEOM,
            ),
            ArticulationObjCfg(
                name="push_button_target",
                usd_path="roboverse_data/assets/rlbench/lamp_off/push_button_target/usd/push_button_target.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/lamp_on/v2/franka_v2.pkl.gz"
    # TODO: add checker
