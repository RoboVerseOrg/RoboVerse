import math

from metasim.cfg.checkers import AndOp, DetectedChecker, RelativeBboxDetector, RotationShiftChecker
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class RotateRedBlockLeftCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = AndOp([
        RotationShiftChecker(
            obj_name="block_red",
            radian_threshold=60 / 180 * math.pi,
        ),
        DetectedChecker(
            obj_name="block_red",
            detector=RelativeBboxDetector(
                base_obj_name="block_red",
                relative_pos=(0.0, 0.0, 0.0),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.1, -0.1, -0.1),
                checker_upper=(0.1, 0.1, 0.1),
                # debug_vis=True,
            ),
        ),
    ])
