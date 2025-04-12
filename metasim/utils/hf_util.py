"""This file contains the utility functions for automatically checking the access and downloading files from the huggingface dataset."""

import os
from multiprocessing import Pool

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from loguru import logger as log

from .parse_util import extract_mesh_paths_from_mjcf, extract_mesh_paths_from_urdf

REPO_ID = "RoboVerseOrg/roboverse_data"
LOCAL_DIR = "roboverse_data"

hf_api = HfApi()
hf_fs = HfFileSystem()


def check_and_download(filepath: str):
    """Check if the file exists in the local directory, and download it from the huggingface dataset if it doesn't exist. If the file is a URDF or MJCF file, it will download the referenced mesh and texture files recursively.

    Args:
        filepath: the filepath to check and download.
    """
    check_and_download_single(filepath)
    if filepath.endswith(".urdf"):
        mesh_paths = extract_mesh_paths_from_urdf(filepath)
        with Pool(processes=16) as p:
            p.map(check_and_download_single, mesh_paths)
    elif filepath.endswith(".xml"):
        mesh_paths = extract_mesh_paths_from_mjcf(filepath)
        with Pool(processes=16) as p:
            p.map(check_and_download_single, mesh_paths)


def check_and_download_single(filepath: str):
    """Check if the file exists in the local directory, and download it from the huggingface dataset if it doesn't exist.

    Args:
        filepath: the filepath to check and download.
    """
    local_exists = os.path.exists(filepath)
    hf_exists = hf_fs.exists(os.path.join("datasets", REPO_ID, os.path.relpath(filepath, LOCAL_DIR)))

    if local_exists:
        ## In this case, the runner is a developer who has the file in their local machine.
        if hf_exists:
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
        except Exception as e:
            raise e
