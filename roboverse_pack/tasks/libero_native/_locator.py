"""Resolve vendored LIBERO assets/MJCF from a local roboverse_data clone or HF.

The native LIBERO tasks reference their scene MJCF + every mesh/texture by a
stable relpath under the ``libero/`` prefix of the ``RoboVerseOrg/roboverse_data``
HF dataset (laid out by ``scripts/native/migrate_libero_assets.py``)::

    libero/tasks/<suite>/<task>.xml         portable MJCF (file= are asset relpaths)
    libero/tasks/<suite>/<task>.goal.json   resolved BDDL goal + OSC config
    libero/assets/robosuite/<...>           Franka / gripper / arena meshes (shared)
    libero/assets/libero/<...>              chiliocosm objects / textures

Resolution prefers a local ``roboverse_data`` clone (dev fast path / offline) and
falls back to downloading the single file from HF on demand — so a clone-less
machine works without the ``libero`` package. The local-clone path imports nothing
heavy (keeps the native task ``mujoco``+``numpy`` only); the HF path lazily imports
``metasim.utils.hf_util`` (never ``libero``/``robosuite``).
"""

from __future__ import annotations

import os
from pathlib import Path

_PREFIX = "libero"  # top-level dir inside the roboverse_data dataset


def _data_dirs():
    """Candidate ``roboverse_data`` roots, most specific first."""
    cands = []
    env = os.environ.get("ROBOVERSE_DATA_DIR")
    if env:
        cands.append(Path(env).expanduser())
    repo_root = Path(__file__).resolve().parents[3]
    cands += [
        Path.cwd() / "roboverse_data",
        repo_root / "roboverse_data",
        repo_root.parent / "roboverse_data",  # sibling clone (~/projects/RoboVerse/roboverse_data)
        Path.home() / "projects" / "RoboVerse" / "roboverse_data",
    ]
    return cands


def libero_data(relpath: str) -> str:
    """Resolve ``libero/<relpath>`` to a concrete local file (clone or HF download).

    Args:
        relpath: path under the dataset's ``libero/`` prefix, e.g.
            ``tasks/libero_object/<task>.xml`` or
            ``assets/robosuite/grippers/meshes/panda_gripper/finger.stl``.

    Returns:
        An existing local filesystem path.
    """
    rel = f"{_PREFIX}/{relpath}"
    for d in _data_dirs():
        p = d / rel
        if p.exists():
            return str(p)
    # cold (clone-less) machine: pull the single file from HF into LOCAL_DIR
    from metasim.utils.hf_util import LOCAL_DIR, check_and_download_single

    target = os.path.join(LOCAL_DIR, rel)
    check_and_download_single(target)
    return target


def libero_asset(relpath: str) -> str:
    """Resolve an asset relpath (the ``file=`` value inside a vendored task MJCF)."""
    return libero_data(f"assets/{relpath}")
