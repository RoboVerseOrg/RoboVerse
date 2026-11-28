"""The task inventory page is generated from the registry, and the committed copy matches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "docs" / "source" / "dataset_benchmark" / "tasks" / "generate_task_docs.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generate_task_docs", GEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_task_docs"] = module
    spec.loader.exec_module(module)
    return module


def test_pack_of_names_the_pack_or_the_example_pack(gen):
    assert gen.pack_of("roboverse_pack.tasks.maniskill.pick_cube") == "maniskill"
    assert gen.pack_of("roboverse_pack.tasks.mjlab.cartpole.balance") == "mjlab"
    assert gen.pack_of("metasim.example.example_pack.tasks.reach_wrappers") == "metasim example pack"
    assert gen.pack_of("roboverse_pack.tasks.task_template") == "task_template"


def test_committed_inventory_matches_the_registry(gen):
    page = gen.render()
    assert (GEN.parent / "inventory.md").read_text() == page, (
        "run python docs/source/dataset_benchmark/tasks/generate_task_docs.py"
    )
    assert "## Suites" in page and "`tabletop-mujoco` / `core`: `pick_cube`" in page
    assert "| `pick_cube` | `roboverse_pack.tasks.maniskill" in page
    assert "## Retired" in page and "task rows retired on 2026-11-27" in page
