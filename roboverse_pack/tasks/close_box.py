from __future__ import annotations

import math

from metasim.example.example_pack.tasks.checkers import JointPosChecker
from metasim.scenario.objects import ArticulationObjCfg as ScenarioArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_box", "close_box", "franka.close_box")
class CloseBoxTask(RLBenchTask):
    """RLBench task for closing a box using a robotic arm."""

    episode_length = 250
    scenario = ScenarioCfg(
        objects=[
            ScenarioArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            )
        ],
        robots=["franka"],
    )
    traj_filepath = "metasim/data/quick_start/trajs/rlbench/close_box/v2"
    checker = JointPosChecker(
        obj_name="box_base", joint_name="box_joint", mode="le", radian_threshold=-14 / 180 * math.pi
    )
