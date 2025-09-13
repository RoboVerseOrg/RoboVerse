from __future__ import annotations

import math

from metasim.example.example_pack.tasks.checkers import JointPosChecker
from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_box", "close_box")
class CloseBoxTask(RLBenchTask):
    max_episode_steps = 250
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
                urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
                mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_box/v2/franka_v2.pkl.gz"
    checker = JointPosChecker(
        obj_name="box_base", joint_name="box_joint", mode="le", radian_threshold=-14 / 180 * math.pi
    )
