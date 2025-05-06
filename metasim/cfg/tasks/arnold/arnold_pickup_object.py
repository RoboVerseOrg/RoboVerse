# ruff: noqa: F401

import os

import numpy as np

from metasim.cfg.checkers import _PositionShiftCheckerWithTolerance
from metasim.cfg.objects import RigidObjCfg
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
json_filenames = sorted(os.listdir("roboverse_data/assets/arnold/pickup_object_merged_pickle"))

# Generate a dictionary of config classes keyed by their class name.
for filename in json_filenames:
    object_id = filename.split("_")[0]
    scale = filename.split("_")[1].strip("[]")
    scale = [float(val) for val in scale.split(",")]
    target = filename.split("_")[2]
    name = f"ArnoldPickupObject{object_id}{get_scale_friendly_name(scale)}Target{target}Cfg"
    code = f"""@configclass
class {name}(ArnoldTaskCfg):
    episode_length = 1600
    try_add_table = False
    decimation = 1

    traj_filepath = "roboverse_data/assets/arnold/pickup_object_merged_pickle/{filename}"
    object_usd_path = "roboverse_data/assets/arnold/sample/custom/Bottle/{object_id}/mobility.usd"
    scale = {scale}
    target = {target}

    def __post_init__(self):
        scale = [s * 0.01 for s in self.scale]

        # cm to m
        target = float(self.target) * 0.01

        self.objects = [
            RigidObjCfg(
                name="Bottle",
                usd_path=self.object_usd_path,
                physics=PhysicStateType.RIGIDBODY,
                scale=scale,
            ),
        ]

        self.checker = _PositionShiftCheckerWithTolerance(
            obj_name="Bottle",
            axis="z",
            distance=target,
            tolerance=0.03,
        )

        self.robot = FrankaCfg(
            name="franka",
        )
"""
    exec(code)


# Now, if you list __all__, you can include all the generated names so that
# "from .arnold.arnold_pickup_object import *" only imports these.
__all__ = [name for name in globals() if name.startswith("ArnoldPickupObject")]
