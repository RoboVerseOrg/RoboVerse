from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import PositionShiftChecker
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.pick_cube", "pick_cube")
class PickCubeTask(ManiskillBaseTask):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="cube",
                usd_path="roboverse_pack/whale_doll/whale_doll.usd",
                urdf_path="roboverse_pack/whale_doll.urdf",
                physics=PhysicStateType.RIGIDBODY,
                enabled_gravity=True,
                fix_base_link=False,
                scale=(0.3, 0.3, 0.3),
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
