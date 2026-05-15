"""``pick_place.functions`` placeholder unblocks the two tasks that
import it."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.general
def test_functions_module_is_importable():
    mod = importlib.import_module("roboverse_pack.tasks.pick_place.functions")
    assert hasattr(mod, "__all__")
    assert mod.__all__ == []


@pytest.mark.general
@pytest.mark.parametrize(
    "task_module",
    [
        "roboverse_pack.tasks.pick_place.approach_grasp_ceramic_teapot",
        "roboverse_pack.tasks.pick_place.approach_grasp_knife",
    ],
)
def test_tasks_that_wildcard_import_functions_load(task_module):
    """The two tasks that use ``from .functions import *`` must import
    cleanly — they're picked up by the task-discovery scan, and a
    ``ModuleNotFoundError`` from one of them silently drops every
    sibling task that hasn't been imported yet."""
    importlib.import_module(task_module)


@pytest.mark.general
def test_registered_task_names_for_those_modules():
    """Both tasks should land in the registry once their modules import."""
    import roboverse_pack.tasks.pick_place.approach_grasp_ceramic_teapot
    import roboverse_pack.tasks.pick_place.approach_grasp_knife  # noqa: F401
    from metasim.task.registry import TASK_REGISTRY

    assert "pick_place.approach_grasp_knife" in TASK_REGISTRY
    assert "pick_place.approach_grasp_hu" in TASK_REGISTRY
