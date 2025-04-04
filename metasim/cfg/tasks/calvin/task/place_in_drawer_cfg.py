## TODO:
## 1. This task is hard because at the beginning, the gripper may hold the block in the air


from metasim.cfg.checkers import DetectedChecker, OrOp, RelativeBboxDetector
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

drawer_detector = RelativeBboxDetector(
    base_obj_name="table",
    relative_pos=(0.118, -0.30, 0.40),
    relative_quat=(1.0, 0.0, 0.0, 0.0),
    checker_lower=(-0.25, -0.125, -0.05),
    checker_upper=(0.25, 0.125, 0.05),
    # debug_vis=True,
)


@configclass
class PlaceInDrawerCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = OrOp([
        DetectedChecker(
            obj_name="block_pink",
            detector=drawer_detector,
            ignore_if_first_check_success=True,
        ),
        DetectedChecker(
            obj_name="block_blue",
            detector=drawer_detector,
            ignore_if_first_check_success=True,
        ),
        DetectedChecker(
            obj_name="block_red",
            detector=drawer_detector,
            ignore_if_first_check_success=True,
        ),
    ])
