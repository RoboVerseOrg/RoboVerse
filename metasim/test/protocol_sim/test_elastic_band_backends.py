from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from metasim.protocol_sim.core.elastic_band import ElasticBandAssist, ElasticBandConfig
from metasim.protocol_sim.core.types import SimRobotObservation


class _DummyIsaacSimArticulation:
    def __init__(self):
        self.body_names = ["g1/base_link", "g1/torso_link"]
        self.calls = []

    def set_external_force_and_torque(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _DummyIsaacSimArticulationLegacy(_DummyIsaacSimArticulation):
    def set_external_force_and_torque(self, *args, **kwargs):
        # Mimic older signatures that reject body_ids/env_ids kwargs.
        if "body_ids" in kwargs or "env_ids" in kwargs:
            raise TypeError("legacy signature")
        self.calls.append((args, kwargs))


class _DummyIsaacSimHandler:
    def __init__(self, robot_name: str, num_envs: int = 2, articulation=None):
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        if articulation is None:
            articulation = _DummyIsaacSimArticulation()
        self._articulation = articulation
        self.scene = type("Scene", (), {"articulations": {robot_name: articulation}})()


class _DummyNewtonModel:
    def __init__(self):
        self.body_key = ["g1/base_link", "g1/torso_link"]


class _DummyNewtonHandler:
    def __init__(self):
        self._model = _DummyNewtonModel()
        self.calls = []

    def _get_body_indices(self, env_id: int, robot_name: str):
        assert env_id == 0
        assert robot_name == "g1_dof29"
        return [0, 1]

    def set_external_body_force(self, body_id: int, force):
        self.calls.append((int(body_id), np.asarray(force, dtype=np.float32)))


class _DummyMjBody:
    def __init__(self, name: str, body_id: int):
        self.name = name
        self.id = body_id


class _DummyMjModel:
    def __init__(self):
        self._bodies = [_DummyMjBody("g1/base_link", 0), _DummyMjBody("g1/torso_link", 1)]
        self.nbody = len(self._bodies)

    def body(self, key):
        if isinstance(key, int):
            return self._bodies[key]
        for body in self._bodies:
            if body.name == key:
                return body
        raise KeyError(key)


class _DummyMujocoHandler:
    def __init__(self):
        self.physics = type(
            "Physics",
            (),
            {
                "model": _DummyMjModel(),
                "data": type("Data", (), {"xfrc_applied": np.zeros((2, 6), dtype=np.float32)})(),
            },
        )()
        self.mj_objects = {"g1_dof29": type("Obj", (), {"model": "g1"})()}


class _DummyIsaacGymApi:
    ENV_SPACE = "env_space"


class _DummyIsaacGymTorch:
    @staticmethod
    def unwrap_tensor(tensor):
        return tensor


class _DummyIsaacGym:
    def __init__(self):
        self.calls = []

    def apply_rigid_body_force_tensors(self, sim, forces, torques, space):
        self.calls.append((sim, forces.clone(), torques.clone(), space))


class _DummyIsaacGymHandler:
    def __init__(self):
        self.gym = _DummyIsaacGym()
        self.sim = object()
        self._rigid_body_states = torch.zeros((4, 13), dtype=torch.float32)
        self._env_rigid_body_global_indices = [{"robot": {"torso_link": 2, "base_link": 1}}]


def _dummy_obs() -> SimRobotObservation:
    # root_state: pos=(0,0,0), quat=(1,0,0,0), lin_vel=(0,0,0), ang_vel=(0,0,0)
    return SimRobotObservation(
        joint_names_sorted=["j0"],
        q_sorted=np.zeros((1,), dtype=np.float32),
        dq_sorted=np.zeros((1,), dtype=np.float32),
        tau_sorted=None,
        root_state=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        body_names_sorted=None,
        body_state=None,
    )


@pytest.mark.general
def test_elastic_band_isaacsim_backend_applies_force_to_selected_body():
    handler = _DummyIsaacSimHandler(robot_name="g1_dof29", num_envs=2)
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler._articulation.calls) == 1
    _, kwargs = handler._articulation.calls[0]
    forces = kwargs["forces"]
    torques = kwargs["torques"]
    assert forces.shape == (2, 2, 3)
    assert torques.shape == (2, 2, 3)
    assert torch.allclose(forces[:, 0, :], torch.zeros((2, 3), dtype=torch.float32))
    assert torch.allclose(forces[:, 1, :], torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]], dtype=torch.float32))
    assert torch.allclose(torques, torch.zeros((2, 2, 3), dtype=torch.float32))


@pytest.mark.general
def test_elastic_band_newton_backend_uses_handler_external_force_hook():
    handler = _DummyNewtonHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=5.0,
            damping=0.0,
            point=(0.0, 0.0, 2.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler.calls) == 1
    body_id, force = handler.calls[0]
    assert body_id == 1
    np.testing.assert_allclose(force, np.array([0.0, 0.0, 10.0], dtype=np.float32), rtol=0.0, atol=1e-6)


@pytest.mark.general
def test_elastic_band_runtime_tuning_updates_length_and_height():
    handler = _DummyNewtonHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 2.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)
    _, force_0 = handler.calls[-1]
    np.testing.assert_allclose(force_0, np.array([0.0, 0.0, 20.0], dtype=np.float32), rtol=0.0, atol=1e-6)

    assist.set_length(1.0)
    assist.set_anchor_height(1.5)
    assert assist.get_length() == pytest.approx(1.0)
    assert assist.get_anchor_height() == pytest.approx(1.5)

    # Length is clamped at 0 to avoid unintuitive negative rest-length behavior.
    assist.set_length(-0.3)
    assert assist.get_length() == pytest.approx(0.0)
    assist.set_length(1.0)

    assist.apply(_dummy_obs(), dt=0.01)
    _, force_1 = handler.calls[-1]
    # distance=1.5, length=1.0, stiffness=10 -> force magnitude 5 on +z
    np.testing.assert_allclose(force_1, np.array([0.0, 0.0, 5.0], dtype=np.float32), rtol=0.0, atol=1e-6)


@pytest.mark.general
def test_elastic_band_manual_release_disables_force_immediately():
    handler = _DummyNewtonHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)
    _, force_before = handler.calls[-1]
    np.testing.assert_allclose(force_before, np.array([0.0, 0.0, 10.0], dtype=np.float32), rtol=0.0, atol=1e-6)

    assist.start_release()
    _, force_release = handler.calls[-1]
    np.testing.assert_allclose(force_release, np.array([0.0, 0.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)

    assist.apply(_dummy_obs(), dt=0.01)
    # After release, apply() exits early and does not issue any new force writes.
    assert len(handler.calls) == 2


@pytest.mark.general
def test_elastic_band_newton_backend_is_tension_only_when_slack():
    handler = _DummyNewtonHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=2.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler.calls) == 1
    _, force = handler.calls[0]
    np.testing.assert_allclose(force, np.array([0.0, 0.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)


@pytest.mark.general
def test_elastic_band_mujoco_backend_writes_xfrc_applied():
    handler = _DummyMujocoHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=6.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    np.testing.assert_allclose(handler.physics.data.xfrc_applied[1, 0:3], np.array([0.0, 0.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(handler.physics.data.xfrc_applied[1, 3:6], np.array([0.0, 0.0, 0.0], dtype=np.float32))


@pytest.mark.general
def test_elastic_band_mujoco_backend_is_tension_only_when_slack():
    handler = _DummyMujocoHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=2.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    np.testing.assert_allclose(handler.physics.data.xfrc_applied[1, 0:3], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(handler.physics.data.xfrc_applied[1, 3:6], np.array([0.0, 0.0, 0.0], dtype=np.float32))


@pytest.mark.general
def test_elastic_band_isaacgym_backend_applies_force_tensor(monkeypatch: pytest.MonkeyPatch):
    dummy_isaacgym_module = types.SimpleNamespace(gymapi=_DummyIsaacGymApi, gymtorch=_DummyIsaacGymTorch)
    monkeypatch.setitem(sys.modules, "isaacgym", dummy_isaacgym_module)

    handler = _DummyIsaacGymHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=7.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler.gym.calls) == 1
    _, forces, torques, space = handler.gym.calls[0]
    assert space == _DummyIsaacGymApi.ENV_SPACE
    assert forces.shape == (4, 3)
    assert torques.shape == (4, 3)
    assert torch.allclose(forces[2], torch.tensor([0.0, 0.0, 7.0], dtype=torch.float32))
    assert torch.allclose(torques[2], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))


@pytest.mark.general
def test_elastic_band_isaacgym_backend_is_tension_only_when_slack(monkeypatch: pytest.MonkeyPatch):
    dummy_isaacgym_module = types.SimpleNamespace(gymapi=_DummyIsaacGymApi, gymtorch=_DummyIsaacGymTorch)
    monkeypatch.setitem(sys.modules, "isaacgym", dummy_isaacgym_module)

    handler = _DummyIsaacGymHandler()
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=2.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler.gym.calls) == 1
    _, forces, torques, _ = handler.gym.calls[0]
    assert torch.allclose(forces[2], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(torques[2], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))


@pytest.mark.general
def test_elastic_band_isaacsim_backend_falls_back_for_legacy_signature():
    articulation = _DummyIsaacSimArticulationLegacy()
    handler = _DummyIsaacSimHandler(robot_name="g1_dof29", num_envs=1, articulation=articulation)
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=4.0,
            damping=0.0,
            point=(0.0, 0.0, 2.0),
            length=0.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(articulation.calls) == 1
    _, kwargs = articulation.calls[0]
    # Legacy fallback path still uses keyword arguments but without body_ids/env_ids.
    assert "body_ids" not in kwargs
    assert "env_ids" not in kwargs
    forces = kwargs["forces"]
    assert forces.shape == (1, 2, 3)
    assert torch.allclose(forces[0, 1, :], torch.tensor([0.0, 0.0, 8.0], dtype=torch.float32))


@pytest.mark.general
def test_elastic_band_isaacsim_backend_is_tension_only_when_slack():
    handler = _DummyIsaacSimHandler(robot_name="g1_dof29", num_envs=1)
    assist = ElasticBandAssist(
        handler=handler,
        robot_name="g1_dof29",
        cfg=ElasticBandConfig(
            stiffness=10.0,
            damping=0.0,
            point=(0.0, 0.0, 1.0),
            length=2.0,
            body_name="torso_link",
            fallback_body_name="base_link",
        ),
    )

    assist.apply(_dummy_obs(), dt=0.01)

    assert len(handler._articulation.calls) == 1
    _, kwargs = handler._articulation.calls[0]
    forces = kwargs["forces"]
    torques = kwargs["torques"]
    assert torch.allclose(forces[0, 1, :], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
    assert torch.allclose(torques[0, 1, :], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
