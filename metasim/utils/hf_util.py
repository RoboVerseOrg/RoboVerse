"""This file contains the utility functions for automatically checking the access and downloading files from the huggingface dataset."""

from __future__ import annotations

import os
import re
from multiprocessing import Pool

import portalocker
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger as log

from metasim.scenario.objects import (
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveFrameCfg,
    PrimitiveMultiBoxCfg,
    PrimitiveSphereCfg,
)
from metasim.scenario.scene import SceneCfg

from .parse_util import extract_mesh_paths_from_urdf, extract_paths_from_mjcf

## This is to avoid circular import
try:
    from metasim.scenario.scenario import ScenarioCfg
except ImportError:
    pass

REPO_ID = "RoboVerseOrg/roboverse_data"
DATA_REPO_IDS = (REPO_ID, "SomaStacksOrg/roboverse_data")


def _resolve_local_dir() -> str:
    """Resolve the asset-cache root to an absolute path.

    Resolved once at import so the cache location doesn't drift with the
    process CWD between calls. ``ROBOVERSE_DATA_DIR`` overrides the
    default ``./roboverse_data`` for shared caches in deployments / CI.
    """
    env_dir = os.environ.get("ROBOVERSE_DATA_DIR")
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return os.path.abspath("roboverse_data")


LOCAL_DIR = _resolve_local_dir()

hf_api = HfApi()


def _find_data_repo(relpath_posix: str, *, is_optional_file: bool = False) -> str | None:
    """Return the first configured data repo containing ``relpath_posix``."""
    errors: list[tuple[str, Exception]] = []
    for repo_id in DATA_REPO_IDS:
        try:
            if hf_api.file_exists(repo_id, relpath_posix, repo_type="dataset"):
                return repo_id
        except Exception as exc:
            errors.append((repo_id, exc))

    if errors:
        msg = "; ".join(f"{repo_id}: {exc}" for repo_id, exc in errors)
        if is_optional_file:
            log.warning(
                f"Optional file {relpath_posix} could not be checked in HuggingFace data sources ({msg}), skipping."
            )
            return None
        raise Exception(f"Could not check {relpath_posix} in HuggingFace data sources ({msg}).") from errors[-1][1]

    return None


def extract_texture_paths_from_mdl(mdl_file_path: str) -> list[str]:
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
        # Pattern: texture_2d("./path/to/texture.png", optional_args) or texture_2d ( "./path", optional_args)
        # Note: Allow optional whitespace before and after opening parenthesis
        texture_pattern = r'texture_2d\s*\(\s*"([^"]+)"[^)]*\)'
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
    if filepath is None:
        log.warning("Received None filepath, skipping download check.")
        return
    local_exists = os.path.exists(filepath)
    if local_exists:
        ## In this case, the runner has the file in their local machine.
        log.info(f"File {filepath} found in local directory.")
        return
    else:
        ## In this case, we didn't find the file in the local directory, the circumstance is complicated.
        # Use POSIX-style paths for the HF dataset API (Windows uses backslashes by default)
        relpath = os.path.relpath(filepath, LOCAL_DIR)
        is_optional_file = filepath.endswith((".mtl", ".png", ".jpg", ".jpeg", ".bmp", ".tga"))
        # A malformed asset descriptor with ``..`` segments would resolve
        # outside LOCAL_DIR — refuse it rather than send the escaped path
        # to the HF API. Optional files warn-and-skip; required files raise.
        if relpath.split(os.sep, 1)[0] == "..":
            msg = (
                f"Refusing to fetch {filepath!r}: resolves outside LOCAL_DIR "
                f"({LOCAL_DIR}) via relative-path traversal ({relpath!r})."
            )
            if is_optional_file:
                log.warning(msg + " Skipping optional file.")
                return
            raise ValueError(msg)
        relpath_posix = relpath.replace(os.sep, "/")
        repo_id = _find_data_repo(relpath_posix, is_optional_file=is_optional_file)

        if repo_id is None:
            if is_optional_file:
                log.warning(f"Optional file {filepath} not found in HuggingFace data sources, skipping.")
                return

            raise Exception(
                f"File {filepath} neither exists in the local directory nor exists in the HuggingFace data sources "
                f"{DATA_REPO_IDS}. Please"
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
                repo_id=repo_id,
                filename=relpath_posix,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
            )
            log.info(f"File {filepath} downloaded from the HuggingFace dataset {repo_id}.")
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
                        texture_relpath = texture_relpath.replace(os.sep, "/")
                        if _find_data_repo(texture_relpath, is_optional_file=True) is not None:
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
                    texture_paths = extract_texture_paths_from_mdl(filepath)
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
            self._add_from_scene(self.scenario.scene)
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

    def _add_from_scene(self, scene: SceneCfg):
        filepath = scene.file_name(self.scenario.simulator)
        if filepath is not None:
            self._add(filepath)

    def _add_from_object(self, obj: BaseObjCfg):
        if (
            isinstance(obj, PrimitiveCubeCfg)
            or isinstance(obj, PrimitiveCylinderCfg)
            or isinstance(obj, PrimitiveMultiBoxCfg)
            or (isinstance(obj, PrimitiveFrameCfg) and obj.file_name(self.scenario.simulator) is None)
            or isinstance(obj, PrimitiveSphereCfg)
        ):
            return

        sim = self.scenario.simulator
        filepath = obj.file_name(sim)
        if filepath is None:
            file_type = obj.file_type[sim]
            raise ValueError(
                f"Object '{obj.name}' has no {file_type} asset path set for the '{sim}' simulator. "
                f"Please set '{file_type}_path' on this object's config. "
                f"Available paths: usd_path={getattr(obj, 'usd_path', None)}, "
                f"urdf_path={getattr(obj, 'urdf_path', None)}, "
                f"mjcf_path={getattr(obj, 'mjcf_path', None)}"
            )
        self._add(filepath)

        for extra_resource in obj.extra_resources:
            self._add(extra_resource)

    def _add(self, filepath: str):
        if filepath is None:
            log.warning("Skipping None filepath in FileDownloader._add")
            return
        self.files_to_download.append(filepath)

    def do_it(self):
        """Download the files specified in the scenario."""
        check_and_download_recursive(self.files_to_download, self.n_processes)
