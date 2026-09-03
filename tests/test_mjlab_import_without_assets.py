"""Importing the mjlab task family must not touch ``roboverse_data``.

Regression: the go1 / yam tasks built their class-level ``scenario`` in the class body, which parses
an MJCF from the asset checkout; ``import roboverse_pack.tasks.mjlab`` therefore failed on any
machine without assets (hosted CI, registry discovery). The scenario is now built on first access.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.general
def test_mjlab_package_imports_without_asset_checkout(tmp_path):
    code = (
        "import roboverse_pack.tasks.mjlab as m, roboverse_pack.tasks.mjlab.mdp.events_dr as dr; "
        "from roboverse_pack.tasks.mjlab.lift_cube_yam_v2 import _YamTaskBase; "
        "assert isinstance(type(_YamTaskBase).__dict__.get('scenario', None), object); print('ok')"
    )
    # An empty cwd: no roboverse_data symlink or checkout anywhere the task code could find.
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ok" in proc.stdout
