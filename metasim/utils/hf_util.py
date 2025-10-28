"""This file contains the utility functions for automatically checking the access and downloading files from the huggingface dataset."""

from __future__ import annotations

import os
import re
from multiprocessing import Pool

import portalocker
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger as log

from metasim.scenario.objects import BaseObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg

from .parse_util import extract_mesh_paths_from_urdf, extract_paths_from_mjcf

## This is to avoid circular import
try:
    from metasim.scenario.scenario import ScenarioCfg
except ImportError:
    pass

REPO_ID = "RoboVerseOrg/roboverse_data"
LOCAL_DIR = "roboverse_data"

hf_api = HfApi()


def _extract_texture_paths_from_mdl(mdl_file_path: str) -> list[str]:
    """Extract texture file paths referenced in an MDL file by parsing its content.

    Args:
        mdl_file_path: Path to the MDL file

    Returns:
        List of absolute texture file paths referenced in the MDL file
    """
    texture_paths = []

    if not os.path.exists(mdl_file_path):
        return texture_paths

    mdl_dir = os.path.dirname(mdl_file_path)

    try:
        with open(mdl_file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse texture_2d declarations in MDL files
        # Pattern: texture_2d("./path/to/texture.png", optional_args)
        texture_pattern = r'texture_2d\("([^"]+)"[^)]*\)'
        matches = re.findall(texture_pattern, content)

        for match in matches:
            if match.strip():  # Skip empty texture declarations
                # Convert relative paths to absolute paths
                if match.startswith("./"):
                    texture_path = os.path.join(mdl_dir, match[2:])  # Remove './'
                elif match.startswith("../"):
                    texture_path = os.path.abspath(os.path.join(mdl_dir, match))
                elif not os.path.isabs(match):
                    texture_path = os.path.join(mdl_dir, match)
                else:
                    texture_path = match

                texture_paths.append(os.path.normpath(texture_path))

    except Exception as e:
        log.debug(f"Failed to parse MDL file {mdl_file_path}: {e}")

    return texture_paths


def check_and_download_single(filepath: str):
    """Check if the file exists in the local directory, and download it from the huggingface dataset if it doesn't exist.

    Args:
        filepath: the filepath to check and download.
    """
    local_exists = os.path.exists(filepath)
    if local_exists:
        ## In this case, the runner has the file in their local machine.
        log.info(f"File {filepath} found in local directory.")
        return
    else:
        ## In this case, we didn't find the file in the local directory, the circumstance is complicated.
        # Use POSIX-style paths for the HF dataset API (Windows uses backslashes by default)
        relpath = os.path.relpath(filepath, LOCAL_DIR)
        relpath_posix = relpath.replace(os.sep, "/")
        hf_exists = hf_api.file_exists(REPO_ID, relpath_posix, repo_type="dataset")

        if not hf_exists:
            if filepath.endswith((".mtl", ".png", ".jpg", ".jpeg", ".bmp", ".tga")):
                log.warning(f"Optional file {filepath} not found in HuggingFace dataset, skipping.")
                return

            raise Exception(
                f"File {filepath} neither exists in the local directory nor exists in the huggingface dataset. Please"
                " report this issue to the developers."
            )

        ## Also, we need to exclude a circumstance that user forgot to update the submodule.
        using_hf_git = os.path.exists(os.path.join(LOCAL_DIR, ".git"))
        if using_hf_git:
            raise Exception(
                "Please update the roboverse_data to the latest version, by running `cd roboverse_data && git pull`."
            )

        ## Finally, download the file from the huggingface dataset.
        try:
            # Ensure the filename uses POSIX separators when requesting from HF hub
            hf_hub_download(
                repo_id=REPO_ID,
                filename=relpath_posix,
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
    os.makedirs(LOCAL_DIR, exist_ok=True)

    lock_path = os.path.join(LOCAL_DIR, "download.lock")
    with portalocker.Lock(lock_path):
        # in parallel env settings, we need to prevent child processes from downloading the same file.

        # check if current process is the main process
        if os.getpid() == os.getppid():
            with Pool(processes=n_processes) as p:
                p.map(check_and_download_single, filepaths)
        else:
            for filepath in filepaths:
                check_and_download_single(filepath)

    new_filepaths = []
    for filepath in filepaths:
        if filepath.endswith(".urdf"):
            mesh_paths = extract_mesh_paths_from_urdf(filepath)
            new_filepaths.extend(mesh_paths)
        elif filepath.endswith(".xml"):
            mesh_paths = extract_paths_from_mjcf(filepath)
            new_filepaths.extend(mesh_paths)
        elif filepath.endswith(".usd") or filepath.endswith(".usda") or filepath.endswith(".usdc"):
            # For USD files, also try to download common texture files
            # USD files often reference textures with relative paths like '../textures/texture_map.png'
            asset_dir = os.path.dirname(filepath)
            # Check for textures directory at the same level as the USD directory
            textures_dir = os.path.join(os.path.dirname(asset_dir), "textures")

            # Try to download common texture file names without listing the entire repo
            try:
                if not os.path.relpath(textures_dir, LOCAL_DIR).startswith(".."):
                    textures_relpath = os.path.relpath(textures_dir, LOCAL_DIR)
                    # Common texture file names to try
                    common_texture_names = [
                        "texture_map.png",
                        "texture.png",
                        "diffuse.png",
                        "albedo.png",
                        "base_color.png",
                    ]
                    for texture_name in common_texture_names:
                        texture_relpath = os.path.join(textures_relpath, texture_name)
                        # Check if this specific file exists on HuggingFace
                        if hf_api.file_exists(REPO_ID, texture_relpath, repo_type="dataset"):
                            texture_path = os.path.join(LOCAL_DIR, texture_relpath)
                            new_filepaths.append(texture_path)
            except Exception as e:
                log.debug(f"Could not check for textures for {filepath}: {e}")
        elif filepath.endswith(".mdl"):
            # For MDL files, parse the file content to extract texture paths
            # This ensures we download exactly what the MDL file references
            if os.path.exists(filepath):
                try:
                    # Parse MDL file and extract texture paths
                    texture_paths = _extract_texture_paths_from_mdl(filepath)
                    # Add textures that don't exist locally to the download list
                    for texture_path in texture_paths:
                        if not os.path.exists(texture_path):
                            new_filepaths.append(texture_path)
                except Exception as e:
                    log.debug(f"Could not parse MDL textures for {filepath}: {e}")

    if len(new_filepaths) > 0:
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
        objects = self.scenario.objects

        for obj in objects:
            self._add_from_object(obj)
        for robot in self.scenario.robots:
            self._add_from_object(robot)
        if self.scenario.scene is not None:
            self._add_from_object(self.scenario.scene)
        # if self.scenario.task is not None:
        #     traj_filepath = self.scenario.task.traj_filepath
        #     if traj_filepath is None:
        #         return

        #     ## HACK: This is hacky
        #     if (
        #         traj_filepath.find(".pkl") == -1
        #         and traj_filepath.find(".json") == -1
        #         and traj_filepath.find(".yaml") == -1
        #         and traj_filepath.find(".yml") == -1
        #     ):
        #         traj_filepath = os.path.join(traj_filepath, f"{self.scenario.robots[0].name}_v2.pkl.gz")
        #     self._add(traj_filepath)

    def _add_from_object(self, obj: BaseObjCfg):
        ## TODO: add a primitive base object class?
        if (
            isinstance(obj, PrimitiveCubeCfg)
            or isinstance(obj, PrimitiveCylinderCfg)
            or isinstance(obj, PrimitiveSphereCfg)
        ):
            return

        if self.scenario.simulator in ["isaaclab", "isaacsim"]:
            self._add(obj.usd_path)
        elif self.scenario.simulator in ["pybullet", "sapien2", "sapien3", "genesis"] or (
            self.scenario.simulator == "isaacgym" and not obj.isaacgym_read_mjcf
        ):
            self._add(obj.urdf_path)
        elif self.scenario.simulator in ["mujoco"] or (
            self.scenario.simulator == "isaacgym" and obj.isaacgym_read_mjcf
        ):
            self._add(obj.mjcf_path)
        elif self.scenario.simulator in ["mjx"]:
            self._add(obj.mjx_mjcf_path)

    def _add(self, filepath: str):
        self.files_to_download.append(filepath)

    def do_it(self):
        """Download the files specified in the scenario."""
        check_and_download_recursive(self.files_to_download, self.n_processes)
