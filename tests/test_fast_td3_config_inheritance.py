"""Regression tests for FastTD3 config loading and ``base.yaml`` inheritance.

Nine task configs (``isaacgym_*`` and half the ``mjx_*``) ship as 3-4 key stubs that
declare ``# Inherits from base.yaml`` but relied on an inheritance mechanism that did not
exist, so every such run died at startup on ``float(cfg("gamma"))`` -> ``float(None)``.
``configs/mjx_walk.yaml`` also had ``headless: flase``, which YAML parses as the truthy
string ``"flase"`` rather than a bool. These tests load every shipped config through the
*real* loader in ``train.py`` and assert each yields all keys ``train.py`` reads without a
default, with correctly-typed values.

``train.py`` cannot be imported wholesale (its module body imports torch/isaacgym, parses
CLI args, and sets env vars at import time), so we extract just ``load_config`` and its
helper from the source via AST and exec them in isolation -- this exercises the shipped
loader code, not a reimplementation.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "roboverse_learn" / "rl" / "fast_td3" / "train.py"
CONFIG_DIR = REPO_ROOT / "roboverse_learn" / "rl" / "fast_td3" / "configs"


def _train_source_tree() -> ast.Module:
    if not TRAIN_PY.exists():
        pytest.skip(f"train.py not found at {TRAIN_PY}")
    return ast.parse(TRAIN_PY.read_text())


def _load_real_loader():
    """Return the actual ``load_config`` from train.py without importing the module body."""
    tree = _train_source_tree()
    wanted = {"_deep_merge", "load_config"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in funcs}
    if missing:
        pytest.fail(f"train.py is missing expected loader function(s): {sorted(missing)}")
    ns = {"os": os, "yaml": yaml, "Any": Any}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(TRAIN_PY), "exec"), ns)
    return ns["load_config"]


def _required_keys() -> set[str]:
    """Keys train.py reads via ``cfg("key")`` with no default -> None would crash it."""
    tree = _train_source_tree()
    with_default: set[str] = set()
    all_keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cfg"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = node.args[0].value
            all_keys.add(key)
            if len(node.args) > 1:
                with_default.add(key)
    required = all_keys - with_default
    assert "gamma" in required, "expected gamma to be a required (no-default) config key"
    return required


ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else []


def test_configs_present():
    assert ALL_CONFIGS, f"no FastTD3 configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=lambda p: p.name)
def test_config_has_all_required_keys(config_path: Path):
    load_config = _load_real_loader()
    required = _required_keys()

    cfg = load_config(str(config_path))

    missing = sorted(required - set(cfg))
    assert not missing, f"{config_path.name} is missing required keys after loading: {missing}"

    # gamma feeds float(cfg("gamma")); a stub without it (or a non-numeric) crashes training.
    assert isinstance(cfg["gamma"], float), f"{config_path.name}: gamma should be float, got {cfg['gamma']!r}"

    # headless must be a real bool -- the 'flase' typo made it the truthy string "flase".
    assert isinstance(cfg["headless"], bool), (
        f"{config_path.name}: headless should be bool, got {type(cfg['headless']).__name__} {cfg['headless']!r}"
    )
