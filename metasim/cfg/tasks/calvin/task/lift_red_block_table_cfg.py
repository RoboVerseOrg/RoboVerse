from metasim.cfg.checkers import DetectedChecker, NotOp, RelativeBboxDetector
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class LiftRedBlockTableCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = NotOp(
        DetectedChecker(
            obj_name="block_red",
            detector=RelativeBboxDetector(
                base_obj_name="block_red",
                relative_pos=(0.0, 0.0, 0.0),
                relative_quat=(1.0, 0.0, 0.0, 0.0),
                checker_lower=(-0.1, -0.1, -0.03),
                checker_upper=(0.1, 0.1, 0.03),
                # debug_vis=True,
            ),
        )
    )
