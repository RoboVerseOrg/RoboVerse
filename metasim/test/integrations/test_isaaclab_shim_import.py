from __future__ import annotations

import importlib

import pytest

from metasim.integrations.isaaclab.shim import ensure_isaaclab_shim


@pytest.mark.general
def test_isaaclab_shim_allows_importing_beyondmimic_cfg():
    # The BeyondMimic IsaacLab configs import `isaaclab.*` modules for config/MDP utilities.
    # In lightweight environments (no USD/pxr), the real IsaacLab package may be partially
    # importable but fail on submodules. The shim should make these configs importable.
    ensure_isaaclab_shim()
    mod = importlib.import_module("roboverse_pack.tasks.beyondmimic.isaaclab.configs.tracking_env_cfg")
    assert hasattr(mod, "TrackingEnvCfg")

    # Also ensure configs that import IsaacLab actuator utilities remain importable.
    # (BeyondMimic's flat cfg imports delayed actuator definitions.)
    flat = importlib.import_module("roboverse_pack.tasks.beyondmimic.isaaclab.configs.flat_env_cfg")
    assert hasattr(flat, "G1FlatEnvCfg")

    # And ensure the config can actually be instantiated under the shim (many tasks call `.replace(...)`).
    cfg = flat.G1FlatEnvCfg()
    assert hasattr(cfg.scene, "robot")
