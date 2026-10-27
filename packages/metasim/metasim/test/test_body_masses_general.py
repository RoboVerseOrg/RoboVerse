"""``BaseSimHandler.get_body_masses`` / ``set_body_masses``: ordering and validation, with no simulator.

Bodies are reported and accepted in ``get_body_names`` order (alphabetical, as in the state) while the
backend hooks work in the engine's order; a column read and written back unchanged keeps the engine's
float64 value; bad input fails before anything reaches the backend.
"""

from __future__ import annotations

import pytest
import torch

from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.sim.base import BaseSimHandler

pytestmark = pytest.mark.general

_NATIVE = ["shin", "base", "thigh"]  # engine order; sorted: base, shin, thigh


class _Handler(BaseSimHandler):
    def __init__(self, num_envs: int = 2):
        self._num_envs = num_envs
        self.object_dict = {
            "leg": ArticulationObjCfg(name="leg", mjcf_path="leg.xml"),
            "cube": PrimitiveCubeCfg(name="cube", size=(0.1, 0.1, 0.1), color=[0.5, 0.5, 0.5]),
        }
        self.masses = torch.tensor([[0.3000000001, 2.0, 1.1]] * num_envs, dtype=torch.float64)
        self.writes: list = []

    def _get_body_names(self, obj_name, sort=True):
        return sorted(_NATIVE) if sort else list(_NATIVE)

    def _get_body_masses(self, obj_name):
        return self.masses.clone()

    def _set_body_masses(self, obj_name, masses, env_ids, scale_inertia):
        self.writes.append((masses.clone(), list(env_ids), scale_inertia))
        self.masses[env_ids] = masses

    def _set_states(self, states, env_ids=None):
        raise NotImplementedError

    def _set_dof_targets(self, actions):
        raise NotImplementedError

    def _get_states(self, env_ids=None):
        raise NotImplementedError

    def _simulate(self):
        raise NotImplementedError


def test_get_reports_bodies_in_get_body_names_order():
    h = _Handler()
    masses = h.get_body_masses("leg")
    assert h.get_body_names("leg") == ["base", "shin", "thigh"]
    assert masses.dtype == torch.float32 and masses.shape == (2, 3)
    torch.testing.assert_close(masses[0], torch.tensor([2.0, 0.3000000001, 1.1], dtype=torch.float32))


def test_set_writes_native_order_for_the_given_envs():
    h = _Handler()
    m = h.get_body_masses("leg")[1:]
    m[0, 2] = 5.0  # thigh
    h.set_body_masses("leg", m, env_ids=[1])
    (written, env_ids, scale), *_ = h.writes
    assert env_ids == [1] and scale is True
    # native order shin, base, thigh; the untouched columns keep the engine's float64 values exactly
    assert written.dtype == torch.float64
    assert written[0].tolist() == [0.3000000001, 2.0, 5.0]
    assert h.masses[0].tolist() == [0.3000000001, 2.0, 1.1]


def test_set_rejects_bad_input_before_the_backend():
    h = _Handler()
    good = h.get_body_masses("leg")
    cases = [
        (dict(obj_name="cube", masses=good), "articulations only"),
        (dict(obj_name="nope", masses=good), "articulations only"),
        (dict(obj_name="leg", masses=good[:, :2]), "shape"),
        (dict(obj_name="leg", masses=-good), "non-negative"),
        (dict(obj_name="leg", masses=torch.full_like(good, float("nan"))), "finite"),
        (dict(obj_name="leg", masses=good[:1], env_ids=[2]), "out of range"),
    ]
    for kwargs, msg in cases:
        with pytest.raises(ValueError, match=msg):
            h.set_body_masses(**kwargs)
    with pytest.raises(ValueError, match="articulations only"):
        h.get_body_masses("cube")
    assert h.writes == []


def test_scaling_the_inertia_of_a_massless_body_is_refused():
    h = _Handler(num_envs=1)
    h.masses[0, 1] = 0.0  # base
    m = h.get_body_masses("leg")
    h.set_body_masses("leg", m)  # unchanged: fine
    m[0, 0] = 1.0
    with pytest.raises(ValueError, match=r"zero-mass bodies \['base'\]"):
        h.set_body_masses("leg", m)
    h.set_body_masses("leg", m, scale_inertia=False)
    assert h.masses[0, 1] == 1.0


def test_a_backend_without_the_hooks_says_so():
    class _Bare(_Handler):
        _get_body_masses = BaseSimHandler._get_body_masses
        _set_body_masses = BaseSimHandler._set_body_masses

    with pytest.raises(NotImplementedError, match="_Bare does not expose body masses"):
        _Bare().get_body_masses("leg")
