"""Tests for the MetaSim-native (zero-dependency) SimplerEnv registration.

The registration test runs anywhere and must NOT need the cloned ``simpler_env`` /
``mani_skill2_real2sim`` packages. The make/reset/step smoke test is gated on the migrated
``roboverse_data`` assets being present (they ship via HF / a roboverse_data checkout).
"""

from __future__ import annotations

import importlib.util

import pytest


def _assets_present() -> bool:
    try:
        from roboverse_pack.tasks.simpler_env._native._assets import data_root

        data_root()
        return True
    except Exception:
        return False


sapien_available = importlib.util.find_spec("sapien") is not None
assets_present = _assets_present()


@pytest.mark.general
def test_metasim_registers_all_25_without_clone():
    """All 25 SimplerEnv ids register via the MetaSim-native registry, no cloned packages needed."""
    import gymnasium as gym

    from roboverse_pack.tasks.simpler_env._metasim.registry import (
        ENVIRONMENTS,
        register_metasim_simpler_tasks,
    )

    ids = register_metasim_simpler_tasks(prefix="SimplerEnv/")
    assert len(ids) == 25
    assert set(ENVIRONMENTS) == {i.removeprefix("SimplerEnv/") for i in ids}
    for t in ENVIRONMENTS:
        assert f"SimplerEnv/{t}" in gym.envs.registry


@pytest.mark.general
def test_native_env_path_has_zero_upstream_imports():
    """No module under ``_native`` or ``_metasim`` may import the cloned simpler_env / clone fork."""
    import pathlib

    offenders = []
    for pkg in ("roboverse_pack.tasks.simpler_env._native", "roboverse_pack.tasks.simpler_env._metasim"):
        root = pathlib.Path(importlib.util.find_spec(pkg).origin).parent
        for f in root.rglob("*.py"):
            for ln in f.read_text().splitlines():
                s = ln.strip()
                if s.startswith(("import ", "from ")) and ("mani_skill2_real2sim" in s or "simpler_env" in s):
                    offenders.append(f"{f.name}: {s}")
    assert not offenders, f"native code imports the clone: {offenders}"


@pytest.mark.skipif(not (sapien_available and assets_present), reason="sapien / roboverse_data assets absent")
def test_metasim_make_reset_step():
    """Smoke-test the MetaSim-native gym.make + reset + step for one task (zero upstream)."""
    import gymnasium as gym
    import numpy as np

    from roboverse_pack.tasks.simpler_env._metasim.registry import register_metasim_simpler_tasks

    register_metasim_simpler_tasks(prefix="SimplerEnv/")
    env = gym.make("SimplerEnv/google_robot_pick_coke_can")
    obs, info = env.reset(seed=0)
    assert "image" in obs and "overhead_camera" in obs["image"]
    obs, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
    assert "success" in info
    env.close()
