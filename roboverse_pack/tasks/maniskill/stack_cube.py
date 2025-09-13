from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg

from .maniskill_base import ManiskillBaseTask


# @register_task("maniskill.stack_cube", "stack_cube")
# registed in metasim/example
class StackCubeTask(ManiskillBaseTask):
    """Stack a red cube on top of a blue base cube and release it."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCubeCfg(
                name="base",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.0, 0.0, 1.0),
            ),
        ],
        robots=["franka"],
    )

    checker = DetectedChecker(
        obj_name="cube",
        detector=RelativeBboxDetector(
            base_obj_name="base",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )

    max_episode_steps = 250

    traj_filepath = "roboverse_data/trajs/maniskill/stack_cube/v2/franka_v2.pkl.gz"
