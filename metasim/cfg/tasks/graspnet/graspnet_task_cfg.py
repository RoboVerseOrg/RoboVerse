import json
import sys

from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass


@configclass
class GraspnetTaskCfg(BaseTaskCfg):
    """The base class for GraspNet tasks."""

    source_benchmark = BenchmarkType.GRASPNET
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True


CONFIG_JSON_PATH = "metasim/cfg/tasks/graspnet/task_config.json"


def _dynamic_import_task(task_name):
    with open(CONFIG_JSON_PATH) as f:
        all_config_data = json.load(f)

    if task_name not in all_config_data:
        raise ValueError(f"Task {task_name} not found in {CONFIG_JSON_PATH}!")

    config_data = all_config_data[task_name]

    task_objects = [
        RigidObjCfg(
            name=obj_name,
            usd_path=f"roboverse_data/assets/graspnet/COMMON/{obj_name}/usd/{obj_name}.usd",
            physics=PhysicStateType.RIGIDBODY,
        )
        for obj_name in config_data["objects"]
    ]

    target_obj_name = config_data.get("target_obj")

    @configclass
    class DynamicGraspNetTaskCfg(GraspnetTaskCfg):
        episode_length = 100
        objects = task_objects
        traj_filepath = f"roboverse_data/trajs/graspnet/grasp_{target_obj_name}/v2/trajectory-franka-{str(config_data['setting_number']).zfill(4)}_v2.pkl"
        checker = PositionShiftChecker(
            obj_name=target_obj_name,
            distance=0.01,
            axis="z",
        )

    DynamicGraspNetTaskCfg.__name__ = f"{task_name}Cfg"

    module_name = "metasim.cfg.tasks.graspnet"
    setattr(sys.modules[module_name], DynamicGraspNetTaskCfg.__name__, DynamicGraspNetTaskCfg)

    return DynamicGraspNetTaskCfg
