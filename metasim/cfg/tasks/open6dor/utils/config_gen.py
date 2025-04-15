import os

# Template for the Python file
template = """

@configclass
class {class_name}Cfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.OPEN6DOR
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    objects = [
        PrimitiveCubeCfg(
            name="table",
            mass=1,
            color=[255, 255, 255],
            size=[0.6, 0.8, 0.3],
        ),
        {objects}
    ]
    traj_filepath = "{traj_filepath}"
"""


# Generate object definitions
def generate_object_def(obj):
    return f"""RigidObjCfg(
            name="{obj["name"]}",
            urdf_path="{obj["urdf_path"]}",
            physics=PhysicStateType.RIGIDBODY
        )"""


import glob
import json
import pickle


def dict_to_pkl(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# CAT = "task_refine_rot_only"
# CAT_TAG = "Rot"


# CAT = "task_refine_pos"
# CAT_TAG = "Pos"


CAT = "task_refine_6dof"
CAT_TAG = "PosRot"
ROOT = f"/home/haoran/Project/RoboVerse/RoboVerse/Open6DOR/Open6DOR/assets/tasks/{CAT}"
tasks = glob.glob(f"{ROOT}/*/*/*/task_config_new5.json")


def to_camel_case(s):
    # Split the string by underscores and capitalize each part
    parts = s.split("_")
    # Keep the first part unchanged and capitalize the rest
    return parts[0].capitalize() + "".join(word.capitalize() for word in parts[1:])


current = 0
total = len(tasks)
for task_i, task in enumerate(tasks):
    current += 1
    print(f"{current}/{total}")
    with open(task) as f:
        config = json.load(f)
    obj_names = config["selected_obj_names"]
    obj_urdfs = config["selected_urdfs"]
    instruction = config["instruction"]
    init_obj_pos = config["init_obj_pos"]
    if CAT_TAG == "Rot":
        CateTag = to_camel_case(config["rotation_instruction_label"].replace(" ", "_"))
        ObjTag = to_camel_case(config["target_obj_name"].replace(" ", "_"))
    elif CAT_TAG == "Pos":
        CateTag = to_camel_case(config["position_tag"].replace(" ", "_"))
        ObjTag = to_camel_case(config["target_obj_name"].replace(" ", "_"))
    elif CAT_TAG == "PosRot":
        PosTag = to_camel_case(config["position_tag"].replace(" ", "_"))
        CateTag = to_camel_case(config["position_tag"].replace(" ", "_")) + to_camel_case(
            config["rotation_instruction_label"].replace(" ", "_")
        )
        ObjTag = to_camel_case(config["target_obj_name"].replace(" ", "_"))
    else:
        raise ValueError(f"Invalid CAT_TAG: {CAT_TAG}")
    # Generate Python files
    output_dir = f"metasim/cfg/tasks/open6dor/task/{CAT_TAG}"
    os.makedirs(output_dir, exist_ok=True)

    class_name = f"Opensdor{CAT_TAG}{CateTag}{ObjTag}{task_i}"
    cfg = {
        "file_name": f"{PosTag}.py",
        "objects": [
            {
                "name": obj_names[i],
                "urdf_path": "data_isaaclab/assets/open6dor/" + obj_urdfs[i],
            }
            for i in range(len(obj_names))
        ],
    }
    tag4 = task.split("/")[-4]
    tag3 = task.split("/")[-3]
    tag2 = task.split("/")[-2]
    root_new = f"data_isaaclab/source_data/open6dor/{CAT}/{tag4}/{tag3}/{tag2}"
    traj_filepath = f"{root_new}/trajectory-unified_wo_traj_v2.pkl"

    # for cfg in config_list:
    object_defs = ",\n        ".join([generate_object_def(obj) for obj in cfg["objects"]])

    file_content = template.format(
        class_name=class_name,
        objects=object_defs,
        traj_filepath=traj_filepath,
    )
    if not os.path.exists(os.path.join(output_dir, cfg["file_name"])):
        with open(os.path.join(output_dir, cfg["file_name"]), "a") as file:
            file_prefix = """import math

from metasim.cfg.checkers import DetectedChecker, OrOp, RelativeBboxDetector
from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass
"""
            file.write(file_prefix)

    with open(os.path.join(output_dir, cfg["file_name"]), "a") as file:
        file.write(file_content)

# print(f"Generated {len(config_list)} configuration files in '{output_dir}' directory.")
