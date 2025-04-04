## TODO:
## 1. The checker is not exactly same as the source simulator
##    In source simulator, we need to check specifically the block that gripper contacts with at the beginning
##    In this implementation, we check if any one block is stacked on any other block

from itertools import product

from metasim.cfg.checkers import DetectedChecker, NotOp, OrOp, RelativeBboxDetector
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

stack_on_red_detector = RelativeBboxDetector(
    base_obj_name="block_red",
    relative_pos=(0.0, 0.0, 0.05),
    relative_quat=(1.0, 0.0, 0.0, 0.0),
    checker_lower=(-0.025, -0.025, -0.025),
    checker_upper=(0.025, 0.025, 0.025),
    ignore_base_ori=True,
    # debug_vis=True,
)

stack_on_blue_detector = RelativeBboxDetector(
    base_obj_name="block_blue",
    relative_pos=(0.0, 0.0, 0.05),
    relative_quat=(1.0, 0.0, 0.0, 0.0),
    checker_lower=(-0.025, -0.025, -0.025),
    checker_upper=(0.025, 0.025, 0.025),
    ignore_base_ori=True,
    # debug_vis=True,
)

stack_on_pink_detector = RelativeBboxDetector(
    base_obj_name="block_pink",
    relative_pos=(0.0, 0.0, 0.05),
    relative_quat=(1.0, 0.0, 0.0, 0.0),
    checker_lower=(-0.025, -0.025, -0.025),
    checker_upper=(0.025, 0.025, 0.025),
    ignore_base_ori=True,
    # debug_vis=True,
)


@configclass
class UnstackBlockCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = NotOp(
        OrOp([
            DetectedChecker(
                obj_name=obj_name,
                detector=detector,
            )
            for obj_name, detector in product(
                ["block_red", "block_blue", "block_pink"],
                [stack_on_red_detector, stack_on_blue_detector, stack_on_pink_detector],
            )
            if obj_name != detector.base_obj_name
        ])
    )
