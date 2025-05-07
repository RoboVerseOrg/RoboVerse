"""This script ensures the USD files are in a clean state.

Specifically, for rigid objects (non-articulation), this script ensures:
1. Remove articulation root API
2. Ensure collision API, with convexDecomposition by default
3. Remove fixed joints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class Args:
    tasks: list[str]
    collision_mode: Literal["convexDecomposition", "convexHull", "meshSimplification"] = "convexDecomposition"


args = tyro.cli(Args)

########################################################
## Launch IsaacLab
########################################################
import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_isaaclab = parser.parse_args([])
args_isaaclab.headless = True
app_launcher = AppLauncher(args_isaaclab)
simulation_app = app_launcher.app

########################################################
## Normal Code
########################################################
from loguru import logger as log
from pxr import Usd, UsdPhysics

from metasim.cfg.objects import RigidObjCfg
from metasim.utils.setup_util import get_task


def is_articulation(usd_path: str):
    joint_count = 0
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
            joint_count += 1
    return joint_count > 0


def remove_articulation_root_api(usd_path: str):
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    stage.Save()


def has_collision_api(usd_path: str):
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def ensure_collision_api(
    usd_path: str,
    collision_mode: Literal["convexDecomposition", "convexHull", "meshSimplification"] = "convexDecomposition",
):
    stage = Usd.Stage.Open(usd_path)
    if has_collision_api(usd_path):
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
                meshCollisionAPI.CreateApproximationAttr().Set(collision_mode)
        stage.Save()
    else:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.ApplyAPI(UsdPhysics.CollisionAPI)
                meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
                meshCollisionAPI.CreateApproximationAttr().Set(collision_mode)
        stage.Save()


def remove_fixed_joint(usd_path: str):
    stage = Usd.Stage.Open(usd_path)
    prim_to_remove = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            prim_to_remove.append(prim)
    for prim in prim_to_remove:
        stage.RemovePrim(prim.GetPrimPath())
    stage.Save()


def main():
    usd_paths = []
    for task in args.tasks:
        task_cfg = get_task(task)
        for obj_cfg in task_cfg.objects:
            if isinstance(obj_cfg, RigidObjCfg) and obj_cfg.usd_path is not None and obj_cfg.usd_path not in usd_paths:
                usd_paths.append(obj_cfg.usd_path)

    for usd_path in usd_paths:
        log.info(f"Cleaning {usd_path}")
        assert not is_articulation(usd_path), f"{usd_path} is an articulation"
        remove_articulation_root_api(usd_path)
        ensure_collision_api(usd_path, args.collision_mode)
        remove_fixed_joint(usd_path)


if __name__ == "__main__":
    main()
