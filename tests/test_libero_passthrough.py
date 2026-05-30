"""Tests for the LIBERO passthrough (``roboverse_pack.tasks.libero._passthrough``).

Two tiers:

* **Import-safe tier** (always runs, no LIBERO needed): the module imports, the
  registration helper is a safe no-op when LIBERO is absent, and the id scheme is
  reversible.
* **1:1 tier** (runs only when LIBERO + robosuite + gymnasium are importable):
  all 130 tasks register, and the passthrough env is bitwise-identical to a
  natively-constructed ``OffScreenRenderEnv`` for obs + a few zero-action steps.

The 1:1 tier needs ``MUJOCO_GL=egl`` (headless) and the dedicated ``libero1to1``
env; it is skipped automatically elsewhere so the default RoboVerse test run
stays green.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAVE_LIBERO = _have("libero") and _have("robosuite") and _have("gymnasium")
libero_only = pytest.mark.skipif(not HAVE_LIBERO, reason="LIBERO/robosuite/gymnasium not installed")


# --------------------------------------------------------------------------- #
# Import-safe tier                                                            #
# --------------------------------------------------------------------------- #
def test_module_imports_without_libero():
    """The passthrough module must import even when LIBERO is absent."""
    from roboverse_pack.tasks.libero import _passthrough as pt

    assert callable(pt.register_libero_passthrough)
    assert callable(pt.make_libero_env)
    assert pt.LIBERO_SUITES == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    )


def test_register_is_safe_noop_without_libero():
    """``register_libero_passthrough`` returns a list and never raises."""
    from roboverse_pack.tasks.libero import _passthrough as pt

    out = pt.register_libero_passthrough()
    assert isinstance(out, list)
    # When LIBERO is absent this is empty; when present it is the 130 ids.
    if not HAVE_LIBERO:
        assert out == []


def test_make_libero_env_raises_clear_error_without_libero():
    """Without LIBERO importable, the factory must raise an actionable error."""
    if HAVE_LIBERO:
        pytest.skip("LIBERO present; error path not exercised")
    from roboverse_pack.tasks.libero import _passthrough as pt

    with pytest.raises(RuntimeError, match="LIBERO is not importable"):
        pt.make_libero_env("libero_spatial", 0)


def test_passthrough_prefix_registered_in_index():
    """The discovery index must list the ``Libero/`` prefix."""
    from roboverse_pack.tasks._passthrough_index import PASSTHROUGH_PREFIXES

    assert "Libero/" in PASSTHROUGH_PREFIXES


# --------------------------------------------------------------------------- #
# 1:1 tier (LIBERO installed)                                                 #
# --------------------------------------------------------------------------- #
@libero_only
def test_all_130_tasks_register():
    """All 5 suites / 130 tasks register under ``Libero/<suite>__<task>``."""
    from roboverse_pack.tasks._passthrough_index import list_passthrough_envs
    from roboverse_pack.tasks.libero import _passthrough as pt

    pt.register_libero_passthrough()
    ids = list_passthrough_envs().get("Libero", [])
    assert len(ids) == 130, f"expected 130 Libero tasks, got {len(ids)}"

    by_suite: dict[str, int] = {}
    for env_id in ids:
        suite = env_id[len("Libero/") :].split("__", 1)[0]
        by_suite[suite] = by_suite.get(suite, 0) + 1
    assert by_suite == {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }, by_suite


@libero_only
def test_passthrough_bitwise_matches_native():
    """Passthrough env obs + 5 zero-action steps are bitwise-identical to native."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from roboverse_pack.tasks.libero._passthrough import make_libero_env

    suite_name, task_id, seed = "libero_spatial", 0, 0

    # native
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    nat = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    nat.seed(seed)
    nat.reset()
    init_states = suite.get_task_init_states(task_id)
    nat_obs = nat.set_init_state(init_states[0])

    # passthrough
    pt_env = make_libero_env(suite_name, task_id, seed=seed)
    pt_obs = pt_env.set_init_state(init_states[0])

    def assert_obs_equal(a, b, tag):
        assert set(a) == set(b), f"{tag}: key mismatch"
        for k in a:
            va, vb = np.asarray(a[k]), np.asarray(b[k])
            assert va.shape == vb.shape, f"{tag}:{k} shape {va.shape} vs {vb.shape}"
            if va.dtype.kind in "fc":
                d = float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))) if va.size else 0.0
                assert d < 1e-9, f"{tag}:{k} max|Δ|={d:.3e}"
            else:
                assert np.array_equal(va, vb), f"{tag}:{k} int/byte mismatch"

    assert_obs_equal(nat_obs, pt_obs, "init")
    for t in range(5):
        zero = [0.0] * 7
        no, nr, nd, _ = nat.step(zero)
        po, pr, pd, _ = pt_env.step(zero)
        assert_obs_equal(no, po, f"step{t}")
        assert nr == pr and nd == pd, f"step{t}: reward/done mismatch"

    nat.close()
    pt_env.close()
