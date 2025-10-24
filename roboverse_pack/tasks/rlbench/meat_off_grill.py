from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.meat_off_grill", "meat_off_grill", "franka.meat_off_grill")
class MeatOffGrillTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="grill_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/grill_visual/usd/grill_visual.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="chicken_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/chicken_visual/usd/chicken_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="steak_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/steak_visual/usd/steak_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/meat_off_grill/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.meat_on_grill", "meat_on_grill", "franka.meat_on_grill")
class MeatOnGrillTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="grill_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/grill_visual/usd/grill_visual.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="chicken_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/chicken_visual/usd/chicken_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="steak_visual",
                usd_path="roboverse_data/assets/rlbench/meat_off_grill/steak_visual/usd/steak_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/meat_on_grill/v2/franka_v2.pkl.gz"
    # TODO: add checker
