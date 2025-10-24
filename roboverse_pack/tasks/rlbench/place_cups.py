from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.place_cups", "place_cups", "franka.place_cups")
class PlaceCupsTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="place_cups_holder_base",
                usd_path="roboverse_data/assets/rlbench/place_cups/place_cups_holder_base/usd/place_cups_holder_base.usd",
                physics=PhysicStateType.XFORM,
            ),
            RigidObjCfg(
                name="mug_visual0",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual1",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual2",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual3",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/place_cups/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.remove_cups", "remove_cups", "franka.remove_cups")
class RemoveCupsTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="place_cups_holder_base",
                usd_path="roboverse_data/assets/rlbench/place_cups/place_cups_holder_base/usd/place_cups_holder_base.usd",
                physics=PhysicStateType.XFORM,
            ),
            RigidObjCfg(
                name="mug_visual0",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual1",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual2",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="mug_visual3",
                usd_path="roboverse_data/assets/rlbench/place_cups/mug_visual1/usd/mug_visual1.usd",  # reuse same asset
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/remove_cups/v2/franka_v2.pkl.gz"
    # TODO: add checker
