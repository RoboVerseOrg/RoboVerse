"""Native LIBERO task is libero/robosuite-free + checker geometry is correct."""

from __future__ import annotations

import importlib
import sys

import numpy as np


def test_native_modules_import_without_libero_or_robosuite():
    before = set(sys.modules)
    importlib.import_module("roboverse_pack.tasks.libero_native.checker")
    importlib.import_module("roboverse_pack.tasks.libero_native.osc")
    importlib.import_module("roboverse_pack.tasks.libero_native.native_task")
    pulled = {m.split(".")[0] for m in set(sys.modules) - before}
    assert "libero" not in pulled, "native task must not import libero"
    assert "robosuite" not in pulled, "native task must not import robosuite"


def test_in_box_matches_libero_semantics():
    from roboverse_pack.tasks.libero_native.checker import in_box

    pos, mat, half = np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1])
    assert in_box(pos, mat, half, np.array([0.0, 0.0, 0.0]))  # center -> inside
    assert not in_box(pos, mat, half, np.array([0.2, 0.0, 0.0]))  # outside x
    # lower-z is extended by 1cm (verbatim from LIBERO SiteObject.in_box)
    assert in_box(pos, mat, half, np.array([0.0, 0.0, -0.105]))
    assert not in_box(pos, mat, half, np.array([0.0, 0.0, -0.12]))
