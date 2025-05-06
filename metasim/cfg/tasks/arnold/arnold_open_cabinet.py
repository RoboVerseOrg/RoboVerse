# ruff: noqa: F401

import gzip
import os
import pickle

import numpy as np

from metasim.cfg.checkers import _JointPosPercentShiftChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.cfg.robots import FrankaCfg
from metasim.cfg.scenes import SceneCfg
from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .arnold_task_cfg import ArnoldTaskCfg

# Global table to store unique scale tuples.
SCALE_TABLE = []


def get_scale_friendly_name(scale_values):
    """Given a list of scale values, return a friendly name based on an index in the SCALE_TABLE.
    If the scale tuple is not in the table, add it.
    """
    scale_tuple = tuple(scale_values)
    if scale_tuple not in SCALE_TABLE:
        SCALE_TABLE.append(scale_tuple)
    index = SCALE_TABLE.index(scale_tuple)
    return f"Scale{index}"


# Example list of JSON filenames.
json_filenames = sorted(os.listdir("roboverse_data/assets/arnold/open_cabinet_merged_pickle"))

for filename in json_filenames:
    object_id = filename.split("_")[0]
    scale = filename.split("_")[1].strip("[]")
    scale = [float(val) for val in scale.split(",")]
    target = filename.split("_")[2]
    targetname = int(float(target) * 100)
    name = f"ArnoldOpenCabinet{object_id}{get_scale_friendly_name(scale)}Target{targetname}Cfg"
    code = f"""@configclass
class {name}(ArnoldTaskCfg):
    episode_length = 2000
    try_add_table = False
    decimation = 1
    scale = {scale}
    target = {target}

    traj_filepath = "roboverse_data/assets/arnold/open_cabinet_merged_pickle/{filename}"
    object_usd_path = "roboverse_data/assets/arnold/sample/StorageFurniture/{object_id}/mobility.usd"

    def __post_init__(self):
        with gzip.open(f"roboverse_data/assets/arnold/open_cabinet_merged_pickle/{filename}", "rb") as f:
            data = pickle.load(f)
        self.joint_name = next(iter(data["franka"][0]["init_state"]["Cabinet"]["dof_pos"].keys()))
        scale = [s * 0.01 for s in self.scale]

        # cm to m
        target = float(self.target) * 0.01

        self.objects = [
            ArticulationObjCfg(
                name="Cabinet",
                usd_path=self.object_usd_path,
                scale=scale,
            ),
        ]
        # assuming uniform scaling
        self.checker = _JointPosPercentShiftChecker(
            obj_name="Cabinet",
            joint_name=self.joint_name,
            percentage_target=self.target,
            threshold=0.08,
            scale=100.0 * (1.0/scale[0]),
            type="revolute",
        )

        self.robot = FrankaCfg(
            name="franka",
        )
"""
    exec(code)


# Now, if you list __all__, you can include all the generated names so that
# "from .arnold.arnold_pickup_object import *" only imports these.
__all__ = [name for name in globals() if name.startswith("ArnoldOpenCabinet")]
