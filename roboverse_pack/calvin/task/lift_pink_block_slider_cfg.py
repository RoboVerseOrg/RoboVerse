from metasim.cfg.checkers import DetectedChecker, NotOp, RelativeBboxDetector
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class LiftPinkBlockSliderCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = NotOp(
        DetectedChecker(
            obj_name="block_pink",
            detector=RelativeBboxDetector(
                base_obj_name="block_pink",
                relative_pos=(0.0, 0.0, 0.0),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.2, -0.2, -0.01),
                checker_upper=(0.2, 0.2, 0.01),
                # debug_vis=True,
            ),
        )
    )
