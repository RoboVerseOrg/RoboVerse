"""Tests for the LIBERO-plus passthrough (``roboverse_pack.tasks.libero_plus``).

Two tiers, mirroring ``test_libero_passthrough.py``:

* **Import-safe tier** (always runs, no LIBERO-plus needed): the module imports,
  the registration helper is a safe no-op when LIBERO-plus is absent, the id
  scheme is reversible, and the ``LiberoPlus/`` prefix is in the discovery index.
* **1:1 tier** (runs only when LIBERO-plus + robosuite + gymnasium import): the
  five suites enumerate to the reference perturbation counts (10,120 total); a
  perturbation env is bitwise-identical (state + reward + done) to a natively
  constructed ``OffScreenRenderEnv``; and a Background-Textures perturbation
  actually changes the render vs its base (guards against silent asset fallback).

The 1:1 tier needs ``MUJOCO_GL=egl`` and the dedicated ``liberoplus`` env; it is
skipped automatically elsewhere so the default RoboVerse test run stays green.

Note: LIBERO-plus shares a *global* EGL render context, so the 1:1 tier renders
one env at a time (build -> record -> close) rather than holding two envs open.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAVE_LIBERO = _have("libero") and _have("robosuite") and _have("gymnasium")
liberoplus_only = pytest.mark.skipif(not HAVE_LIBERO, reason="LIBERO-plus/robosuite/gymnasium not installed")


def _is_liberoplus() -> bool:
    """True only when the installed ``libero`` is actually LIBERO-plus (>130 tasks)."""
    if not HAVE_LIBERO:
        return False
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    return pt.suite_task_count("libero_spatial") > 10


# --------------------------------------------------------------------------- #
# Import-safe tier                                                            #
# --------------------------------------------------------------------------- #
def test_module_imports_without_liberoplus():
    """The passthrough module must import even when LIBERO-plus is absent."""
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    assert callable(pt.register_liberoplus_passthrough)
    assert callable(pt.make_liberoplus_env)
    assert callable(pt.make_liberoplus_env_from_bddl)
    assert pt.LIBERO_PLUS_SUITES == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    )


def test_register_is_safe_noop_without_liberoplus():
    """``register_liberoplus_passthrough`` returns a list and never raises."""
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    out = pt.register_liberoplus_passthrough()
    assert isinstance(out, list)
    if not _is_liberoplus():
        assert out == []


def test_make_env_raises_clear_error_without_liberoplus():
    """Without LIBERO-plus importable, the factory must raise an actionable error."""
    if HAVE_LIBERO:
        pytest.skip("libero present; error path not exercised")
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    with pytest.raises(RuntimeError, match="LIBERO-plus is not importable"):
        pt.make_liberoplus_env("libero_spatial", 0)


def test_passthrough_prefix_registered_in_index():
    """The discovery index must list the ``LiberoPlus/`` prefix."""
    from roboverse_pack.tasks._passthrough_index import PASSTHROUGH_PREFIXES

    assert "LiberoPlus/" in PASSTHROUGH_PREFIXES


def test_reference_counts_constant():
    """The reference per-suite counts sum to the documented 10,120-task benchmark."""
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    assert sum(pt.REFERENCE_SUITE_COUNTS.values()) == 10120


# --------------------------------------------------------------------------- #
# 1:1 tier (LIBERO-plus installed)                                            #
# --------------------------------------------------------------------------- #
@liberoplus_only
def test_suites_enumerate_to_reference_counts():
    """All five suites enumerate to the reference perturbation counts."""
    if not _is_liberoplus():
        pytest.skip("installed `libero` is base LIBERO, not LIBERO-plus")
    from roboverse_pack.tasks.libero_plus import _passthrough as pt

    counts = {s: pt.suite_task_count(s) for s in pt.LIBERO_PLUS_SUITES}
    assert counts == pt.REFERENCE_SUITE_COUNTS, counts


@liberoplus_only
def test_passthrough_bitwise_matches_native():
    """A perturbation env is bitwise-identical (state+reward+done) to native.

    Rendered one env at a time (shared global EGL context). Compares all
    non-image obs keys, reward and done across reset + 5 zero-action steps.
    """
    if not _is_liberoplus():
        pytest.skip("installed `libero` is base LIBERO, not LIBERO-plus")
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from roboverse_pack.tasks.libero_plus._passthrough import make_liberoplus_env

    suite_name, seed = "libero_spatial", 0
    # A Camera-Viewpoint perturbation (synthetic descriptor parsed by the wrapper).
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_id = next(
        i for i, n in enumerate(suite.get_task_names()) if "_view_" in n and "_initstate_0" in n and "_noise_" not in n
    )

    def record(env, n=5):
        traj = []
        for _ in range(n):
            obs, rew, done, _ = env.step([0.0] * 7)
            traj.append(({k: np.asarray(v) for k, v in obs.items()}, float(rew), bool(done)))
        return traj

    # passthrough (record, close)
    pt_env = make_liberoplus_env(suite_name, task_id, seed=seed)
    pt_traj = record(pt_env)
    pt_env.close()

    # native (independent construction, record, close)
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    nat = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    nat.seed(seed)
    nat.reset()
    nat_traj = record(nat)
    nat.close()

    for t, ((po, pr, pd), (no, nr, nd)) in enumerate(zip(pt_traj, nat_traj)):
        assert set(po) == set(no), f"step{t}: obs key mismatch"
        for k in po:
            if any(h in k.lower() for h in ("image", "rgb", "depth")):
                continue  # images bucketed (EGL/noise stochasticity); state is the contract
            d = float(np.max(np.abs(po[k].astype(np.float64) - no[k].astype(np.float64)))) if po[k].size else 0.0
            assert d == 0.0, f"step{t}:{k} state max|Δ|={d:.3e}"
        assert pr == nr and pd == nd, f"step{t}: reward/done mismatch"


@liberoplus_only
def test_background_texture_perturbation_is_applied():
    """A Background-Textures variant must render differently from its base scene.

    Guards against silent asset fallback: if the texture asset were missing and
    the env fell back to the base scene, this would (wrongly) read identical.
    """
    if not _is_liberoplus():
        pytest.skip("installed `libero` is base LIBERO, not LIBERO-plus")
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark

    from roboverse_pack.tasks.libero_plus._passthrough import make_liberoplus_env, make_liberoplus_env_from_bddl

    suite_name = "libero_spatial"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    names = suite.get_task_names()
    tid = next(i for i, n in enumerate(names) if "_table_" in n)
    base_stem = names[tid].split("_table_")[0]
    base_path = os.path.join(suite.get_task_bddl_file_path(tid).rsplit("/", 1)[0], base_stem + ".bddl")
    assert os.path.isfile(base_path), f"base BDDL missing: {base_path}"

    def frame(env, n=5):
        obs = None
        for _ in range(n):
            obs, *_ = env.step([0.0] * 7)
        return np.asarray(obs["agentview_image"]).astype(np.float64)

    # one env at a time (shared EGL texture context)
    eb = make_liberoplus_env_from_bddl(base_path, camera_heights=128, camera_widths=128)
    eb.seed(0)
    eb.reset()
    img_base = frame(eb)
    eb.close()

    ep = make_liberoplus_env(suite_name, tid, seed=0)
    img_pert = frame(ep)
    ep.close()

    mae = float(np.abs(img_pert - img_base).mean())
    assert mae > 1.0, f"texture perturbation had no visible effect (MAE={mae:.3f}) — asset fallback?"
