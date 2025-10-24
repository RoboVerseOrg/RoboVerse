from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task(
    "rlbench.slide_cabinet_open_and_place_cups",
    "slide_cabinet_open_and_place_cups",
    "franka.slide_cabinet_open_and_place_cups",
)
class SlideCabinetOpenAndPlaceCupsTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="cabinet_base",
                usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cabinet_base/usd/cabinet_base.usd",
            ),
            RigidObjCfg(
                name="cup_visual",
                usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cup_visual/usd/cup_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/slide_cabinet_open_and_place_cups/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.take_cup_out_from_cabinet", "take_cup_out_from_cabinet", "franka.take_cup_out_from_cabinet")
class TakeCupOutFromCabinetTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="cabinet_base",
                usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cabinet_base/usd/cabinet_base.usd",
            ),
            RigidObjCfg(
                name="cup_visual",
                usd_path="roboverse_data/assets/rlbench/slide_cabinet_open_and_place_cups/cup_visual/usd/cup_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/take_cup_out_from_cabinet/v2/franka_v2.pkl.gz"
    # TODO: add checker
