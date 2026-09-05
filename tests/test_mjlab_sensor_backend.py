"""The mjlab sensors say which backend they can run on instead of failing on the first update.

A sensor picked the Newton path whenever the handler had no ``physics`` attribute, so the parallel
MuJoCo wrapper (``--sim mujoco --num_envs > 1``) and MJX went down that path and raised
``AttributeError`` on the first ``update``, which the manager-based env used to swallow, feeding zeroed
contacts to the reward. The backend is now decided at construction, with the DR events' backend naming
and exception, so a task with both gets one kind of error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from roboverse_pack.tasks.mjlab.mdp.sensors import ContactSensor, ContactSensorCfg, _sensor_backend


def _env(handler, simulator="mujoco"):
    handler.scenario = SimpleNamespace(simulator=simulator)
    return SimpleNamespace(handler=handler, step_dt=0.02, device=torch.device("cpu"), num_envs=2)


class _NoPhysics:  # the parallel MuJoCo wrapper, or MJX: neither physics nor the Newton queries
    pass


def test_parallel_wrapper_and_mjx_are_refused_at_construction_naming_the_backend():
    cfg = ContactSensorCfg(name="feet", primary_bodies=("FL_foot",), secondary_body="floor", fields=("found",))
    with pytest.raises(
        NotImplementedError, match=r"'ContactSensor' is not implemented for the 'mujoco' backend .*_NoPhysics"
    ):
        ContactSensor(_env(_NoPhysics(), "mujoco"), cfg)
    with pytest.raises(NotImplementedError, match="for the 'mjx' backend"):
        ContactSensor(_env(_NoPhysics(), "mjx"), cfg)


def test_backend_detection_prefers_mujoco_then_the_needed_newton_api():
    class _Mujoco:
        physics = object()

    class _Newton:
        def get_net_contact_forces_by_body(self):
            return None

    assert _sensor_backend(_env(_Mujoco()), "ContactSensor", ("get_net_contact_forces_by_body",)) == "mujoco"
    assert _sensor_backend(_env(_Newton(), "newton"), "ContactSensor", ("get_net_contact_forces_by_body",)) == "newton"
    assert _sensor_backend(_env(_NoPhysics()), "TerrainHeightSensor", ()) == "newton", (
        "states-only sensors run anywhere"
    )
    with pytest.raises(NotImplementedError, match="get_subtree_field"):
        _sensor_backend(_env(_Newton(), "newton"), "BuiltinSensor", ("get_subtree_field",))
