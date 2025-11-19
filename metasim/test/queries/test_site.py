"""Integration tests for metasim/queries/site.py using real MuJoCo and MJX."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.queries.site import SitePos, _get_site_id, _site_cache
from metasim.test.queries.conftest import get_query_scenario


def _pick_robot_site_name(handler) -> str:
    """Pick a site name belonging to the robot from the MuJoCo model."""
    import pytest as _pytest

    mj_model = handler.physics.model
    robot_name = handler.robot.name
    prefix = f"{robot_name}/"

    for i in range(mj_model.nsite):
        name = mj_model.site(i).name
        if name.startswith(prefix):
            return name

    _pytest.skip(f"No site with prefix '{prefix}' found in MuJoCo model")


def _pick_mjx_robot_site_name(handler) -> str:
    """Pick a site name belonging to the robot from the MJX MuJoCo model."""
    import pytest as _pytest

    mj_model = handler._mj_model
    robot_name = handler._robot.name
    prefix = f"{robot_name}/"

    for i in range(mj_model.nsite):
        name = mj_model.site(i).name
        if name.startswith(prefix):
            return name

    _pytest.skip(f"No site with prefix '{prefix}' found in MJX MuJoCo model")


def site_id_cache_mujoco_query(handler):
    """Child-process body: validate _get_site_id caching on a real MuJoCo model."""

    _site_cache.clear()
    mj_model = handler.physics.model
    assert mj_model.nsite > 0

    site_name = mj_model.site(0).name
    sid1 = _get_site_id(mj_model, site_name)
    sid2 = _get_site_id(mj_model, site_name)

    assert sid1 == sid2
    key = id(mj_model)
    assert key in _site_cache
    assert _site_cache[key][site_name] == sid1
    logger = log.bind(sim="mujoco")
    logger.info("site-id cache populated correctly for MuJoCo model")


def site_pos_mujoco_query(handler):
    """Child-process body: validate SitePos on a real MuJoCo handler."""
    import torch as _torch

    full_site_name = _pick_robot_site_name(handler)
    site_name = full_site_name.split("/", 1)[1]

    query = SitePos(site_name)
    query.bind_handler(handler)

    pos = query()
    assert isinstance(pos, _torch.Tensor)
    assert pos.shape == (1, 3)

    sid = _get_site_id(handler.physics.model, full_site_name)
    expected = handler.data.site_xpos[sid]
    assert _torch.allclose(pos.squeeze(0), _torch.as_tensor(expected, dtype=pos.dtype), atol=1e-5)


def site_pos_mjx_query(handler):
    """Child-process body: validate SitePos on a real MJX handler."""
    import torch as _torch

    full_site_name = _pick_mjx_robot_site_name(handler)
    site_name = full_site_name.split("/", 1)[1]

    query = SitePos(site_name)
    query.bind_handler(handler)

    pos = query()
    assert isinstance(pos, _torch.Tensor)
    assert pos.shape == (handler.num_envs, 3)


def test_get_site_id_populates_cache_with_real_model(shared_handler):
    """Use shared_handler; run only when sim == 'mujoco'."""
    sim, proxy = shared_handler
    if sim != "mujoco":
        pytest.skip("Skipping MuJoCo SitePos cache test for non-mujoco sim")

    pytest.importorskip("mujoco")
    proxy.run_test(site_id_cache_mujoco_query)


def test_site_pos_mujoco_returns_world_position_tensor(shared_handler):
    """SitePos should return a (1, 3) tensor matching MuJoCo's site_xpos."""
    sim, proxy = shared_handler
    if sim != "mujoco":
        pytest.skip("Skipping MuJoCo SitePos test for non-mujoco sim")

    pytest.importorskip("mujoco")
    proxy.run_test(site_pos_mujoco_query)


def test_site_pos_mjx_returns_world_position_tensor(shared_handler):
    """SitePos should return an (N_env, 3) tensor for MJX."""
    sim, proxy = shared_handler
    if sim != "mjx":
        pytest.skip("Skipping MJX SitePos test for non-mjx sim")

    pytest.importorskip("mujoco")
    pytest.importorskip("jax")
    proxy.run_test(site_pos_mjx_query)


def _get_site_test_funcs(sim: str):
    """Return the site query bodies that should run for the requested simulator."""
    mapping = {
        "mujoco": [
            site_id_cache_mujoco_query,
            site_pos_mujoco_query,
        ],
        "mjx": [site_pos_mjx_query],
    }
    return mapping.get(sim, [])


def _build_site_scenario(sim: str, num_envs: int):
    """Reuse the shared query scenario configuration but allow MJX overrides."""
    if sim == "mjx":
        scenario = get_query_scenario("mujoco", num_envs)
        scenario.update(simulator="mjx")
        return scenario
    return get_query_scenario(sim, num_envs)


def _process_run_handler(scenario, test_funcs):
    """Child-process helper used by run_test() for standalone execution."""
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    try:
        for func in test_funcs:
            log.info(f"[site standalone] Running {func.__name__}()")
            func(handler)
    finally:
        handler.close()


def run_test(sim="mujoco", num_envs=1):
    """Standalone runner to mirror pytest execution for site query tests."""
    import multiprocessing as mp

    log.info(f"Running Site query tests in standalone mode: sim={sim}, num_envs={num_envs}")
    if sim == "mujoco" and num_envs != 1:
        log.warning("MuJoCo only supports num_envs=1; overriding requested value %s -> 1", num_envs)
        num_envs = 1

    test_funcs = _get_site_test_funcs(sim)
    if not test_funcs:
        log.warning(f"No standalone Site query tests registered for sim '{sim}'")
        return

    scenario = _build_site_scenario(sim, num_envs)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_process_run_handler, args=(scenario, test_funcs))
    proc.start()
    proc.join(timeout=90)

    if proc.is_alive():
        proc.terminate()
        raise TimeoutError(f"Standalone Site query test for {sim} (num_envs={num_envs}) timed out")

    assert proc.exitcode == 0, f"Standalone Site query child exited with code {proc.exitcode}"
    log.info("Standalone Site query tests finished successfully.")


if __name__ == "__main__":
    import sys

    sim = sys.argv[1] if len(sys.argv) > 1 else "mujoco"
    num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_test(sim, num_envs)
