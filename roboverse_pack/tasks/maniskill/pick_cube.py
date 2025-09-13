from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import PositionShiftChecker
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.pick_cube", "pick_cube")
class PickCubeTask(ManiskillBaseTask):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            )
        ],
        robots=["franka"],
    )

    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"

    max_episode_steps = 250

    checker = PositionShiftChecker(
        obj_name="cube",
        distance=0.1,
        axis="z",
    )
