"""Shared pytest configuration for the RoboVerse content test suite.

``tests/`` runs in very different environments: a developer workstation with every optional
extra installed and a full ``roboverse_data`` checkout, and a CPU-only CI container that
installs only ``.[dev,mujoco]`` and fetches no assets. Two markers keep that difference
honest instead of red:

    @pytest.mark.requires_optional("zarr", extra="learn")
    @pytest.mark.requires_asset("robots/openarm_wuji/openarm_wuji.xml")

Each skips — naming what is missing and how to get it — when an optional dependency or a
``roboverse_data`` asset is genuinely absent.

The markers are evaluated at test *setup*, so they cover the common case of a test whose
optional import happens inside the test body. When a test *module* imports an optional
dependency at module scope, the import blows up at collection before any marker can run —
use a module-level ``pytest.importorskip("<dep>", reason=...)`` there instead (as
``test_fusion_pipeline.py`` and the ``mani_skill``/``sapien`` suites do).

These markers are NOT a way to quiet a failing test. A test that fails because the code
under test is wrong must stay red: skipping it would hide the defect, which is strictly
worse than a red suite. Only a genuinely absent optional dependency or asset may skip.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_HF_DATA_REPO = "https://huggingface.co/datasets/RoboVerseOrg/roboverse_data"


def roboverse_data_root() -> Path:
    """Return the ``roboverse_data`` asset root: ``$ROBOVERSE_DATA``, else the in-repo default.

    Mirrors how the packs themselves resolve assets (see
    ``roboverse_pack.tasks.simpler_env._native._assets``) minus the HF download step — a test
    must never trigger a multi-GB fetch as a side effect of being collected.
    """
    env_root = os.environ.get("ROBOVERSE_DATA")
    return Path(env_root) if env_root else REPO_ROOT / "roboverse_data"


def missing_assets(relpaths: tuple[str, ...]) -> list[str]:
    """Return the subset of ``relpaths`` (relative to the data root) that is not on disk."""
    root = roboverse_data_root()
    return [relpath for relpath in relpaths if not (root / relpath).exists()]


def missing_modules(names: tuple[str, ...]) -> list[str]:
    """Return the subset of ``names`` that cannot be imported."""
    missing = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def pytest_configure(config: pytest.Config) -> None:
    """Register the RoboVerse-specific markers (also listed in ``pyproject.toml``)."""
    config.addinivalue_line(
        "markers",
        "requires_optional(*modules, extra=None): skip unless the optional dependencies are installed",
    )
    config.addinivalue_line(
        "markers",
        "requires_asset(*relpaths): skip unless the roboverse_data assets are present",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests whose optional dependencies or ``roboverse_data`` assets are absent."""
    for marker in item.iter_markers(name="requires_optional"):
        missing = missing_modules(marker.args)
        if missing:
            extra = marker.kwargs.get("extra")
            how = f'python -m pip install -e ".[{extra}]"' if extra else f"python -m pip install {' '.join(missing)}"
            pytest.skip(f"optional dependency not installed: {', '.join(missing)} — install with: {how}")

    for marker in item.iter_markers(name="requires_asset"):
        missing = missing_assets(marker.args)
        if missing:
            pytest.skip(
                f"roboverse_data asset(s) not fetched: {', '.join(missing)} — "
                f"looked under {roboverse_data_root()}; point $ROBOVERSE_DATA at a "
                f"roboverse_data checkout or fetch them from {_HF_DATA_REPO}"
            )
