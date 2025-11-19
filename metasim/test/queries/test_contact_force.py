"""Integration tests for ContactForces on real simulators."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.queries.contact_force import ContactForces
from metasim.test.queries.conftest import get_query_scenario


def _assert_basic_shapes(handler, query: ContactForces):
    """Shared shape checks for all backends."""
    hist = query.contact_forces_history
    assert isinstance(hist, torch.Tensor)
    assert hist.shape[0] == handler.scenario.num_envs
    assert hist.shape[1] == query.history_length

    sorted_body_names = handler.get_body_names(handler.robots[0].name, True)
    assert hist.shape[2] == len(sorted_body_names)

    current = query.contact_forces
    assert isinstance(current, torch.Tensor)
    assert current.shape == (handler.scenario.num_envs, hist.shape[2], 3)
    assert torch.allclose(current, hist[:, -1])


def contact_forces_isaacsim_query(handler):
    """Child-process body: ContactForces on IsaacSim, cross-check with ContactSensor."""
    query = ContactForces(history_length=3)
    query.bind_handler(handler)
    _assert_basic_shapes(handler, query)

    # IsaacSim branch uses ContactSensor net_forces_w internally; verify consistency.
    sensor_forces = handler.contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    expected = sensor_forces[:, query.body_ids_reindex, :]
    current = query.contact_forces
    assert torch.allclose(current, expected, atol=1e-5)

    log.info("ContactForces matches IsaacSim ContactSensor net_forces_w.")


def contact_forces_isaacgym_query(handler):
    """Child-process body: ContactForces on IsaacGym, cross-check with raw net-contact tensor."""
    query = ContactForces(history_length=3)
    query.bind_handler(handler)
    _assert_basic_shapes(handler, query)

    # IsaacGym branch uses acquire_net_contact_force_tensor; handler keeps a wrapped copy.
    raw = handler._contact_forces  # (num_envs * num_bodies, 3)
    assert raw is not None
    reshaped = raw.view(handler.scenario.num_envs, -1, 3)[:, query.body_ids_reindex, :]

    current = query.contact_forces
    assert torch.allclose(current, reshaped, atol=1e-6)

    log.info("ContactForces matches IsaacGym net_contact_force tensor.")


def contact_forces_mujoco_query(handler):
    """Child-process body: ContactForces on MuJoCo, sanity-check non-zero and balanced forces."""
    # Step the simulator a bit so contacts develop reliably.
    for _ in range(10):
        handler.simulate()

    query = ContactForces(history_length=3)
    query.bind_handler(handler)
    _assert_basic_shapes(handler, query)

    current = query.contact_forces  # (num_envs, n_body, 3)

    # Expect at least some non-zero contact forces once the robot has interacted with the ground.
    assert torch.any(current.norm(dim=-1) > 0), "MuJoCo contact forces should be non-zero for some bodies."

    log.info("ContactForces on MuJoCo produces non-zero yet globally balanced contact forces.")


def test_contact_forces_isaacsim_with_shared_handler(shared_handler):
    """Run ContactForces test using the shared handler process (sim == 'isaacsim')."""
    sim, proxy = shared_handler  # (sim name, HandlerProxy)
    if sim != "isaacsim":
        pytest.skip("Skipping ContactForces test for non-isaacsim sim")

    proxy.run_test(func=contact_forces_isaacsim_query)


def test_contact_forces_isaacgym_with_shared_handler(shared_handler):
    """Run ContactForces test using the shared handler process (sim == 'isaacgym')."""
    sim, proxy = shared_handler
    if sim != "isaacgym":
        pytest.skip("Skipping ContactForces test for non-isaacgym sim")

    pytest.importorskip("isaacgym")
    proxy.run_test(func=contact_forces_isaacgym_query)


def test_contact_forces_mujoco_with_shared_handler(shared_handler):
    """Run ContactForces test using the shared handler process (sim == 'mujoco')."""
    sim, proxy = shared_handler
    if sim != "mujoco":
        pytest.skip("Skipping ContactForces test for non-mujoco sim")

    pytest.importorskip("mujoco")
    proxy.run_test(func=contact_forces_mujoco_query)


def _get_contact_force_test_funcs(sim: str):
    """Return the test bodies that should run for the requested simulator."""
    mapping = {
        "isaacsim": [contact_forces_isaacsim_query],
        "isaacgym": [contact_forces_isaacgym_query],
        "mujoco": [contact_forces_mujoco_query],
    }
    return mapping.get(sim, [])


def _process_run_handler(scenario, test_funcs):
    """Child-process helper used by run_test() for standalone execution."""
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    try:
        for func in test_funcs:
            log.info(f"[contact_forces standalone] Running {func.__name__}()")
            func(handler)
    finally:
        handler.close()


def run_test(sim="isaacsim", num_envs=1):
    """Standalone runner that mirrors the pytest fixture + child-process pipeline."""
    import multiprocessing as mp

    log.info(f"Running ContactForces tests in standalone mode: sim={sim}, num_envs={num_envs}")
    if sim == "mujoco" and num_envs != 1:
        log.warning("MuJoCo only supports num_envs=1; overriding requested value %s -> 1", num_envs)
        num_envs = 1

    test_funcs = _get_contact_force_test_funcs(sim)
    if not test_funcs:
        log.warning(f"No standalone ContactForces tests registered for sim '{sim}'")
        return

    scenario = get_query_scenario(sim, num_envs)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_process_run_handler, args=(scenario, test_funcs))
    proc.start()
    proc.join(timeout=90)

    if proc.is_alive():
        proc.terminate()
        raise TimeoutError(f"Standalone ContactForces test for {sim} (num_envs={num_envs}) timed out")

    assert proc.exitcode == 0, f"Standalone ContactForces child exited with code {proc.exitcode}"
    log.info("Standalone ContactForces tests finished successfully.")


if __name__ == "__main__":
    import sys

    sim = sys.argv[1] if len(sys.argv) > 1 else "isaacsim"
    num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_test(sim, num_envs)
