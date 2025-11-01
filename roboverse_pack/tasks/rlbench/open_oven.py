from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_oven", "open_oven", "franka.open_oven")
class OpenOvenTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="oven_base",
                usd_path="roboverse_data/assets/rlbench/open_oven/oven_base/usd/oven_base.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_oven/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.put_tray_in_oven", "put_tray_in_oven", "franka.put_tray_in_oven")
class PutTrayInOvenTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="oven_base",
                usd_path="roboverse_data/assets/rlbench/open_oven/oven_base/usd/oven_base.usd",
            ),
            RigidObjCfg(
                name="tray_visual",
                usd_path="roboverse_data/assets/rlbench/put_tray_in_oven/tray_visual/usd/tray_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_tray_in_oven/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.take_tray_out_of_oven", "take_tray_out_of_oven", "franka.take_tray_out_of_oven")
class TakeTrayOutOfOvenTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="oven_base",
                usd_path="roboverse_data/assets/rlbench/open_oven/oven_base/usd/oven_base.usd",
            ),
            RigidObjCfg(
                name="tray_visual",
                usd_path="roboverse_data/assets/rlbench/put_tray_in_oven/tray_visual/usd/tray_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/take_tray_out_of_oven/v2/franka_v2.pkl.gz"
    # TODO: add checker
