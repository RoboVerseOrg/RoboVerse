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
    # Step long enough for G1 (default z ≈ 0.78m, dt=0.005s) to fall, land,
    # and develop steady contacts. Free-fall alone is ~80 steps; allow 4×
    # margin for bounce/settle so the assertion is robust.
    for _ in range(400):
        handler.simulate()
    query = ContactForces(history_length=3)
    query.bind_handler(handler)
    # Re-poll once more so the latest snapshot reflects the post-settle state
    # rather than the moment the query happened to be constructed.
    query()
    _assert_basic_shapes(handler, query)

    current = query.contact_forces  # (num_envs, n_body, 3)

    # Expect at least some non-zero contact forces once the robot has interacted with the ground.
    assert torch.any(current.norm(dim=-1) > 0), "MuJoCo contact forces should be non-zero for some bodies."

    # World frame, force on the body: the ground supports the settled robot, so the summed contact
    # force over its bodies points up (+z) and carries its weight. Before the frame rotation and
    # sign fix this sum was in the contact frame with the reaction sign.
    total = current[0].sum(dim=0)
    assert total[2] > 0, f"net contact force on the robot must point up, got {total.tolist()}"
    assert total[2] > 0.5 * total.norm(), f"net contact force is not mostly vertical: {total.tolist()}"

    log.info("ContactForces on MuJoCo produces non-zero yet globally balanced contact forces.")


@pytest.mark.general
def test_mujoco_contact_frame_helper_reports_world_frame_support_force():
    """A cube resting on a 15 degree ramp: the helper gives the cube +m g along world z and the ramp the
    reaction. Summing the raw ``mj_contactForce`` components put ~cos(15 degrees) m g on the contact
    normal and the rest on a tangent axis, and the old signs gave the cube the reaction.
    """
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from metasim.queries.contact_force import mujoco_net_contact_forces_world

    xml = """
    <mujoco><option gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="ramp" type="box" size="5 5 0.05" euler="0 15 0" friction="2 0.005 0.0001"/>
        <body name="cube" pos="0 0 0.25">
          <freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1" friction="2 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.ncon >= 1
    forces = mujoco_net_contact_forces_world(model, data)
    cube = forces[model.body("cube").id]
    assert np.linalg.norm(cube) == pytest.approx(9.81, rel=0.05), cube
    assert cube[2] / np.linalg.norm(cube) == pytest.approx(1.0, abs=0.05), cube
    assert forces[model.body("world").id][2] == pytest.approx(-cube[2], rel=1e-6)
