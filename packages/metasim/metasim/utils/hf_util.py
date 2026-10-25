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
DOWNLOAD_LOCK_TIMEOUT_S = 3600.0
"""How long a process waits for another process's in-flight download of the same asset tree."""
DATA_REPO_IDS = (REPO_ID, "SomaStacksOrg/roboverse_data")

_UPSTREAM_MIRRORS: dict[str, str] = {}
"""``roboverse_data``-relative prefix -> URL base of the upstream that publishes the files under it.

A content pack registers the upstream of an asset it vendors by reference (a Menagerie robot, a mesh set a
manufacturer hosts) with :func:`register_upstream_mirror`. A file missing locally and from every HuggingFace
data repo is then fetched from ``url_base + relpath[len(prefix):]`` into the same place the HuggingFace copy
would land, so a fresh checkout works before (and after) the asset is mirrored into the data repo.
"""


def register_upstream_mirror(prefix: str, url_base: str) -> None:
    """Fetch files under ``roboverse_data/<prefix>`` from ``url_base`` when no data repo has them.

    Args:
        prefix: Directory under ``roboverse_data`` the mirror covers, e.g. ``robots/sharpa_wave/mjcf``.
        url_base: URL that serves the same tree, e.g. a pinned-commit ``raw.githubusercontent.com`` path.
            Pin the upstream revision in the URL so a checkout keeps fetching the bytes the config was
            written against.
    """
    prefix = prefix.strip("/") + "/"
    _UPSTREAM_MIRRORS[prefix] = url_base.rstrip("/") + "/"


def _upstream_url(relpath_posix: str) -> str | None:
    """The upstream URL of ``relpath_posix`` when a registered mirror covers it (the longest prefix wins)."""
    for prefix in sorted(_UPSTREAM_MIRRORS, key=len, reverse=True):
        if relpath_posix.startswith(prefix):
            return _UPSTREAM_MIRRORS[prefix] + relpath_posix[len(prefix) :]
    return None


def _download_from_upstream(url: str, filepath: str) -> None:
    """Stream ``url`` into ``filepath`` through a private temp file in the same directory, published atomically.

    A failed fetch leaves no partial file, and two unlocked callers fetching the same file (``check_and_download_single``
    is also called directly, outside ``download.lock``) cannot interleave their writes.
    """
    import shutil
    import tempfile
    import urllib.request

    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(filepath) + ".", suffix=".part", dir=directory)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


_DESCRIPTION_SUFFIXES = (".xml", ".urdf", ".usd", ".usda", ".usdc")
_COMPANION_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "README.md")
"""Files fetched, when they exist, next to a robot / object description: the asset's licence and notes travel with it."""


def _fetch_companions(filepath: str) -> None:
    """Fetch the licence / README that sit next to a freshly downloaded description file, from wherever it came.

    Each companion is optional: absent upstream, or unreachable, it is skipped without a word beyond debug.
    """
    if not filepath.endswith(_DESCRIPTION_SUFFIXES):
        return
    directory = os.path.dirname(filepath)
    for name in _COMPANION_FILES:
        companion = os.path.join(directory, name)
        if os.path.exists(companion):
            continue
        try:
            relpath_posix = os.path.relpath(os.path.realpath(companion), os.path.realpath(LOCAL_DIR)).replace(
                os.sep, "/"
            )
            repo_id = _find_data_repo(relpath_posix, is_optional_file=True)
            if repo_id is not None:
                hf_hub_download(repo_id=repo_id, filename=relpath_posix, repo_type="dataset", local_dir=LOCAL_DIR)
                continue
            upstream_url = _upstream_url(relpath_posix)
            if upstream_url is not None:
                _download_from_upstream(upstream_url, companion)
        except Exception as exc:  # the asset itself is already in place; its paperwork is best effort
            log.debug(f"companion {name} of {filepath} not fetched: {exc}")


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


_RELATIVE_DATA_PREFIX = "roboverse_data" + os.sep


def _outside_local_dir_message(filepath: str, relpath: str) -> str:
    """Explain why ``filepath`` cannot be fetched into ``LOCAL_DIR``.

    Asset paths in configs are ``roboverse_data/...`` and are opened by the simulator backends
    relative to the *current working directory*; ``ROBOVERSE_DATA_DIR`` only moves where downloads
    land. So a relative path that does not resolve under ``LOCAL_DIR`` is almost always a CWD /
    ``ROBOVERSE_DATA_DIR`` mismatch, not a malicious ``..`` — say so, and point at the asset when it
    already exists in the configured directory.
    """
    if not os.path.isabs(filepath) and filepath.startswith(_RELATIVE_DATA_PREFIX):
        in_local_dir = os.path.join(LOCAL_DIR, filepath[len(_RELATIVE_DATA_PREFIX) :])
        found = f" The asset exists at {in_local_dir!r}." if os.path.exists(in_local_dir) else ""
        return (
            f"Cannot fetch {filepath!r}: asset paths are relative to the working directory "
            f"({os.getcwd()!r}), which is not the parent of LOCAL_DIR ({LOCAL_DIR!r}).{found} "
            f"Run from the repository root, or symlink ./roboverse_data to LOCAL_DIR "
            f"(ROBOVERSE_DATA_DIR only changes where downloads are stored)."
        )
    return (
        f"Refusing to fetch {filepath!r}: resolves outside LOCAL_DIR ({LOCAL_DIR}) via relative-path "
        f"traversal ({relpath!r})."
    )


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
        log.debug(f"File {filepath} found in local directory.")
        return
    else:
        ## In this case, we didn't find the file in the local directory, the circumstance is complicated.
        # Use POSIX-style paths for the HF dataset API (Windows uses backslashes by default)
        # Compare *real* paths: LOCAL_DIR is commonly a symlink to a shared asset checkout (CI caches,
        # git worktrees), and a symlink-vs-target mismatch must not be mistaken for traversal.
        relpath = os.path.relpath(os.path.realpath(filepath), os.path.realpath(LOCAL_DIR))
        is_optional_file = filepath.endswith((".mtl", ".png", ".jpg", ".jpeg", ".bmp", ".tga"))
        # A malformed asset descriptor with ``..`` segments would resolve
        # outside LOCAL_DIR — refuse it rather than send the escaped path
        # to the HF API. Optional files warn-and-skip; required files raise.
        if relpath.split(os.sep, 1)[0] == "..":
            msg = _outside_local_dir_message(filepath, relpath)
            if is_optional_file:
                log.warning(msg + " Skipping optional file.")
                return
            raise ValueError(msg)
        relpath_posix = relpath.replace(os.sep, "/")
        repo_id = _find_data_repo(relpath_posix, is_optional_file=is_optional_file)

        if repo_id is None:
            upstream_url = _upstream_url(relpath_posix)
            if upstream_url is not None:
                try:
                    _download_from_upstream(upstream_url, os.path.join(LOCAL_DIR, relpath))
                except Exception as exc:
                    if is_optional_file:
                        log.warning(
                            f"Optional file {filepath} could not be fetched from {upstream_url} ({exc}), skipping."
                        )
                        return
                    raise Exception(
                        f"File {filepath} is in none of the HuggingFace data sources {DATA_REPO_IDS} and fetching it "
                        f"from its registered upstream {upstream_url} failed: {exc}"
                    ) from exc
                log.info(f"File {filepath} downloaded from its upstream {upstream_url}.")
                _fetch_companions(filepath)
                return
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
        _fetch_companions(filepath)


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
    # portalocker.Lock defaults to a 5 s timeout, after which a sibling process that is merely
    # *waiting* for another process's download raises AlreadyLocked. Parallel-env workers and any
    # two scripts sharing a roboverse_data checkout hit this whenever a fetch takes longer than 5 s
    # (mesh-heavy assets routinely do). Wait for the download instead.
    with portalocker.Lock(lock_path, timeout=DOWNLOAD_LOCK_TIMEOUT_S, check_interval=0.5):
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
