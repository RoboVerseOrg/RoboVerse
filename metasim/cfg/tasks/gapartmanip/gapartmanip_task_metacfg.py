import json
import math
import os.path as osp
import sys

from metasim.cfg.checkers import JointPosShiftChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .gapartmanip_open_box import GAPartManipOpenBoxCfg
from .gapartmanip_open_toilet import GAPartManipOpenToiletCfg

CONFIG_JSON_PATH = "metasim/cfg/tasks/gapartmanip/task_config.json"
TASKS = [
    "OpenBox",
    "OpenToilet",
]


def _dynamic_import_task(task_name):
    with open(CONFIG_JSON_PATH) as f:
        all_config_data = json.load(f)

    if task_name not in all_config_data:
        raise ValueError(f"Task {task_name} not found in {CONFIG_JSON_PATH}!")

    task = None
    for _task in TASKS:
        if _task in task_name:
            task = _task
    if task is None:
        raise ValueError(f"Cannot find a valid task for {task_name} in the available tasks: {TASKS}.")

    config_data = all_config_data[task_name]  # GAPartManip{task_name}{object_id}

    data_dir = (
        f"roboverse_data/assets/gapartmanip/{config_data['object']['category']}/{config_data['object']['model_id']}"
    )
    urdf_path = f"{data_dir}/mobility_annotation_gapartnet_joint_axis_normalized.urdf"
    if not osp.exists(urdf_path):
        urdf_path = f"{data_dir}/mobility_annotation_gapartnet.urdf"
    if not osp.exists(urdf_path):
        raise ValueError(f"URDF file not found at {urdf_path}. Please check the model path.")

    task_objects = [
        ArticulationObjCfg(
            name=config_data["object"]["name"],
            fix_base_link=True,
            urdf_path=urdf_path,
            scale=config_data["object"]["scale"],
        ),
    ]

    task_checker = None
    if task == "OpenBox":
        task_checker = JointPosShiftChecker(
            obj_name=config_data["object"]["name"],
            joint_name=config_data["task"]["target_joint_name"],
            threshold=config_data["task"]["target_joint_delta_degree"] / 180.0 * math.pi,  # Convert degree to radian
        )
    elif task == "OpenToilet":
        task_checker = JointPosShiftChecker(
            obj_name=config_data["object"]["name"],
            joint_name=config_data["task"]["target_joint_name"],
            threshold=config_data["task"]["target_joint_delta_degree"] / 180.0 * math.pi,  # Convert degree to radian
        )

    task_traj_filepath = f"roboverse_data/trajs/gapartmanip/{task.lower()}/v2/trajectory-franka_extension-{config_data['object']['model_id']}_v2.pkl.gz"

    if task == "OpenBox":
        basecfg = GAPartManipOpenBoxCfg
    elif task == "OpenToilet":
        basecfg = GAPartManipOpenToiletCfg

    @configclass
    class DynamicGAPartManipTaskMetaCfg(basecfg):
        episode_length = config_data["task"].get("episode_length", 300)
        objects = task_objects
        traj_filepath = task_traj_filepath
        checker = task_checker

    DynamicGAPartManipTaskMetaCfg.__name__ = f"{task_name}MetaCfg"

    module_name = "metasim.cfg.tasks.gapartmanip"
    setattr(
        sys.modules[module_name],
        DynamicGAPartManipTaskMetaCfg.__name__,
        DynamicGAPartManipTaskMetaCfg,
    )

    return DynamicGAPartManipTaskMetaCfg
