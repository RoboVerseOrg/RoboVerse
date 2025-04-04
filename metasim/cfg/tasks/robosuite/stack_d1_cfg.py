from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

SCALE_FACTOR = 2.0


@configclass
class StackD1Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.ROBOSUITE
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 200
    traj_filepath = "data_isaaclab/source_data/robosuite/stack_d1/stack_d1_zero_height_v2.pkl"
    can_tabletop = True
    objects = [
        PrimitiveCubeCfg(
            name="cubeA",
            size=[0.02 * SCALE_FACTOR, 0.02 * SCALE_FACTOR, 0.02 * SCALE_FACTOR],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="cubeB",
            size=[0.025 * SCALE_FACTOR, 0.025 * SCALE_FACTOR, 0.025 * SCALE_FACTOR],
            mass=0.02,
            physics=PhysicStateType.RIGIDBODY,
            color=[0.0, 1.0, 0.0],
        ),
    ]
    checker = DetectedChecker(
        obj_name="cubeA",
        detector=RelativeBboxDetector(
            base_obj_name="cubeB",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )
