"""Integration tests for ContactForces on real simulators."""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.queries.contact_force import ContactForces


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


@pytest.mark.isaacsim
def test_contact_forces_isaacsim_with_shared_handler(handler):
    """Run ContactForces test using the shared handler process (sim == 'isaacsim')."""
    query = ContactForces(history_length=3)
    query.bind_handler(handler)
    _assert_basic_shapes(handler, query)
    # IsaacSim branch uses ContactSensor net_forces_w internally; verify consistency.
    sensor_forces = handler.contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    expected = sensor_forces[:, query.body_ids_reindex, :]
    current = query.contact_forces
    assert torch.allclose(current, expected, atol=1e-5)

    log.info("ContactForces matches IsaacSim ContactSensor net_forces_w.")


@pytest.mark.isaacgym
def test_contact_forces_isaacgym_with_shared_handler(handler):
    """Run ContactForces test using the shared handler process (sim == 'isaacgym')."""
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


@pytest.mark.mujoco
def test_contact_forces_mujoco_with_shared_handler(handler):
    """Run ContactForces test using the shared handler process (sim == 'mujoco')."""
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
