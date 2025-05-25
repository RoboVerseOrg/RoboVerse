"""This file contains the utility functions for automatically checking the access and downloading files from the huggingface dataset."""

from __future__ import annotations

import os
from multiprocessing import Pool

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from loguru import logger as log

from metasim.cfg.objects import BaseObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg

from .parse_util import extract_mesh_paths_from_mjcf, extract_mesh_paths_from_urdf

## This is to avoid circular import
try:
    from metasim.cfg.scenario import ScenarioCfg
except ImportError:
    pass

REPO_ID = "RoboVerseOrg/roboverse_data"
LOCAL_DIR = "roboverse_data"

hf_api = HfApi()
hf_fs = HfFileSystem()


def _check_and_download_single(filepath: str):
    """Check if the file exists in the local directory, and download it from the huggingface dataset if it doesn't exist.

    Args:
        filepath: the filepath to check and download.
    """
    local_exists = os.path.exists(filepath)
    hf_exists = hf_fs.exists(os.path.join("datasets", REPO_ID, os.path.relpath(filepath, LOCAL_DIR)))

    if local_exists:
        ## In this case, the runner is a developer who has the file in their local machine.
        if hf_exists:
            log.info(f"File {filepath} found in local directory.")
            return
        else:
            log.warning(f"Please upload the file {filepath} to the huggingface dataset!")
            return
    else:
        ## In this case, we didn't find the file in the local directory, the circumstance is complicated.

        ## Make sure the file exists in the huggingface dataset.
        if not hf_exists:
            raise Exception(
                f"File {filepath} neither exists in the local directory nor exists in the huggingface dataset. Please"
                " report this issue to the developers."
            )

        ## Also, we need to exclude a circumstance that a developer forgot to update the submodule.
        using_hf_git = os.path.exists(os.path.join(LOCAL_DIR, ".git"))
        if using_hf_git:
            raise Exception(
                "Please update the roboverse_data to the latest version, by running `cd roboverse_data && git pull`."
            )

        ## Finally, download the file from the huggingface dataset.
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=os.path.relpath(filepath, LOCAL_DIR),
                repo_type="dataset",
                local_dir=LOCAL_DIR,
            )
            log.info(f"File {filepath} downloaded from the huggingface dataset.")
        except Exception as e:
            raise e


def check_and_download_recursive(filepaths: list[str], n_processes: int = 16):
    """Check if the files exist in the local directory, and download them from the huggingface dataset if they don't exist. If the file is a URDF or MJCF file, it will download the referenced mesh and texture files recursively.

    Args:
        filepaths (list[str]): the filepaths to check and download.
        n_processes (int): the number of processes to use for downloading. Default is 16.
    """
    if len(filepaths) == 0:
        return

    with Pool(processes=n_processes) as p:
        p.map(_check_and_download_single, filepaths)

    new_filepaths = []
    for filepath in filepaths:
        if filepath.endswith(".urdf"):
            mesh_paths = extract_mesh_paths_from_urdf(filepath)
            new_filepaths.extend(mesh_paths)
        elif filepath.endswith(".xml"):
            mesh_paths = extract_mesh_paths_from_mjcf(filepath)
            new_filepaths.extend(mesh_paths)
    check_and_download_recursive(new_filepaths, n_processes)


class FileDownloader:
    """Parallel file downloader for the files specified in the scenario.

    Args:
        scenario: the scenario configuration.
        n_processes (int): the number of processes to use for downloading. Default is 16.
    """

    def __init__(self, scenario: ScenarioCfg, n_processes: int = 16):
        self.scenario = scenario
        self.files_to_download = []
        self._add_from_scenario()
        self.n_processes = n_processes

    def _add_from_scenario(self):
        ## TODO: delete this line after scenario is automatically overwritten by task
        objects = self.scenario.task.objects if self.scenario.task is not None else self.scenario.objects

        for obj in objects:
            self._add_from_object(obj)
        for robot in self.scenario.robots:
            self._add_from_object(robot)
        if self.scenario.scene is not None:
            self._add_from_object(self.scenario.scene)
        if self.scenario.task is not None:
            traj_filepath = self.scenario.task.traj_filepath
            if traj_filepath is None:
                return

            ## HACK: This is hacky
            if (
                traj_filepath.find(".pkl") == -1
                and traj_filepath.find(".json") == -1
                and traj_filepath.find(".yaml") == -1
                and traj_filepath.find(".yml") == -1
            ):
                traj_filepath = os.path.join(traj_filepath, f"{self.scenario.robots[0].name}_v2.pkl.gz")
            self._add(traj_filepath)

    def _add_from_object(self, obj: BaseObjCfg):
        ## TODO: add a primitive base object class?
        if (
            isinstance(obj, PrimitiveCubeCfg)
            or isinstance(obj, PrimitiveCylinderCfg)
            or isinstance(obj, PrimitiveSphereCfg)
        ):
            return

        if self.scenario.sim in ["isaaclab"]:
            self._add(obj.usd_path)
        elif self.scenario.sim in ["isaacgym", "pybullet", "sapien2", "sapien3", "genesis"]:
            self._add(obj.urdf_path)
        elif self.scenario.sim in ["mujoco"]:
            self._add(obj.mjcf_path)
        elif self.scenario.sim in ["mjx"]:
            self._add(obj.mjx_mjcf_path)

    def _add(self, filepath: str):
        self.files_to_download.append(filepath)

    def do_it(self):
        """Download the files specified in the scenario."""
        check_and_download_recursive(self.files_to_download, self.n_processes)
