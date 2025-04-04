from metasim.cfg.checkers import (
    AndOp,
    DetectedChecker,
    RelativeBboxDetector,
)
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class StackThreeD0Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.ROBOSUITE
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 300
    traj_filepath = "data_isaaclab/source_data/robosuite/stack_three_d0/stack_three_d0_zero_height_v2.pkl"  # fmt: skip
    can_tabletop = True
    objects = [
        PrimitiveCubeCfg(
            name="cubeA",
            size=[0.02 * 2, 0.02 * 2, 0.02 * 2],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="cubeB",
            size=[0.025 * 2, 0.025 * 2, 0.025 * 2],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[0.0, 1.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="cubeC",
            size=[0.02 * 2, 0.02 * 2, 0.02 * 2],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[0.0, 0.0, 1.0],
        ),
    ]

    checker = AndOp([
        DetectedChecker(
            obj_name="cubeA",
            detector=RelativeBboxDetector(
                base_obj_name="cubeB",
                relative_pos=(0.0, 0.0, 0.04),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.025, -0.02, -0.02),
                checker_upper=(0.025, 0.02, 0.02),
                ignore_base_ori=True,
                fixed=True,
            ),
        ),
        DetectedChecker(
            obj_name="cubeC",
            detector=RelativeBboxDetector(
                base_obj_name="cubeA",
                relative_pos=(0.0, 0.0, 0.04),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.02, -0.02, -0.02),
                checker_upper=(0.02, 0.02, 0.02),
                ignore_base_ori=True,
                fixed=False,
            ),
        ),
    ])
