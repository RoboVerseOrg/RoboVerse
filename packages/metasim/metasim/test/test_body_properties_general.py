"""``BaseSimHandler`` body properties (masses, friction, restitution): ordering and validation, no simulator.

Bodies are reported and accepted in ``get_body_names`` order (alphabetical, as in the state) while the
backend hooks work in the engine's order; a column read and written back unchanged keeps the engine's
float64 value; bad input fails before anything reaches the backend. A property the engine does not model
warns and is NaN instead of raising; a body without collision shapes has NaN friction and ignores writes.
"""

from __future__ import annotations

import pytest
import torch

from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.sim.base import BaseSimHandler
from metasim.utils.log import reset_warn_once

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

    # shin has no collision shape
    friction = None

    def _get_body_friction(self, obj_name):
        if self.friction is None:
            self.friction = torch.tensor([[float("nan"), 0.8, 1.0]] * self.num_envs, dtype=torch.float64)
        return self.friction.clone()

    def _set_body_friction(self, obj_name, friction, env_ids):
        self.writes.append((friction.clone(), list(env_ids)))
        self.friction[env_ids] = friction

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
        (dict(obj_name="cube", masses=good), "shape"),  # a rigid object is one body
        (dict(obj_name="nope", masses=good), "not a robot or object"),
        (dict(obj_name="leg", masses=good[:, :2]), "shape"),
        (dict(obj_name="leg", masses=-good), "non-negative"),
        (dict(obj_name="leg", masses=torch.full_like(good, float("nan"))), "finite"),
        (dict(obj_name="leg", masses=good[:1], env_ids=[2]), "out of range"),
    ]
    for kwargs, msg in cases:
        with pytest.raises(ValueError, match=msg):
            h.set_body_masses(**kwargs)
    with pytest.raises(ValueError, match="not a robot or object"):
        h.get_body_masses("nope")
    assert h.writes == []


def test_a_rigid_object_is_one_body_over_its_engine_bodies():
    # the stub's three engine bodies stand for a compound rigid object
    h = _Handler(num_envs=1)
    assert h.get_body_masses("cube").tolist() == [[pytest.approx(3.4000000001)]]
    assert h.get_body_friction("cube").tolist() == [[pytest.approx(0.9)]]  # mean over the bodies with shapes

    h.set_body_masses("cube", h.get_body_masses("cube").double() * 2)
    (written, env_ids, _), *_ = h.writes
    torch.testing.assert_close(written[0], torch.tensor([0.6000000002, 4.0, 2.2], dtype=torch.float64))  # same shares
    h.set_body_friction("cube", torch.tensor([[0.5]]))
    assert torch.isnan(h.friction[0, 0]) and h.friction[0, 1:].tolist() == [0.5, 0.5]

    masses, friction = h.masses.clone(), h.friction.clone()
    h.set_body_friction("cube", h.get_body_friction("cube"))  # a float32 round trip changes nothing
    h.set_body_masses("cube", h.get_body_masses("cube"))
    assert torch.equal(h.masses, masses) and torch.allclose(h.friction, friction, equal_nan=True, rtol=0, atol=0)

    # a rigid object with no collision shape at all (a static primitive on Newton): warns, does not raise
    h.friction = torch.full((1, 3), float("nan"), dtype=torch.float64)
    h.set_body_friction("cube", torch.tensor([[0.5]]))
    assert torch.isnan(h.friction).all()

    h.masses.zero_()
    with pytest.raises(ValueError, match="3 engine bodies and no mass"):
        h.set_body_masses("cube", torch.tensor([[1.0]]), scale_inertia=False)


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


def test_friction_is_nan_for_a_shapeless_body_and_a_write_there_is_ignored(caplog):
    reset_warn_once()
    h = _Handler()
    f = h.get_body_friction("leg")  # sorted: base, shin, thigh
    assert torch.isnan(f[:, 1]).all() and f[0, 0] == pytest.approx(0.8)
    h.set_body_friction("leg", f)  # round trip, NaN included: nothing is ignored, nothing changes
    assert torch.equal(h.writes[-1][0][:, 1:], h.friction[:, 1:]) and torch.isnan(h.writes[-1][0][:, 0]).all()
    f[:, 1] = 0.5  # shin: no collision shape
    f[:, 2] = 0.3  # thigh
    with pytest.raises(ValueError, match="finite and non-negative"):
        h.set_body_friction("leg", torch.tensor([[-1.0, 0.5, 0.3]] * 2))
    h.set_body_friction("leg", f)
    written = h.writes[-1][0]
    assert torch.isnan(written[:, 0]).all(), "the shapeless body keeps NaN"
    assert written[:, 2].tolist() == pytest.approx([0.3, 0.3])


def test_an_engine_without_restitution_warns_instead_of_raising(caplog):
    reset_warn_once()

    class _NoBounce(_Handler):
        unsupported_body_properties = {"restitution": "This engine has no bounce coefficient."}

    h = _NoBounce()
    assert torch.isnan(h.get_body_restitution("leg")).all()
    h.set_body_restitution("leg", torch.zeros(2, 3))  # validated, warned, nothing written
    assert h.writes == []
    with pytest.raises(ValueError, match="shape"):
        h.set_body_restitution("leg", torch.zeros(2, 2))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):  # validated even though nothing is written
        h.set_body_restitution("leg", torch.full((2, 3), 5.0))


def test_an_empty_env_ids_writes_nothing():
    h = _Handler()
    h.set_body_masses("leg", torch.zeros(0, 3), env_ids=[])
    h.set_body_friction("leg", torch.zeros(0, 3), env_ids=[])
    assert h.writes == []


def test_restitution_must_lie_in_zero_one():
    class _Bouncy(_Handler):
        def _get_body_restitution(self, obj_name):
            return torch.zeros(self.num_envs, 3, dtype=torch.float64)

        def _set_body_restitution(self, obj_name, restitution, env_ids):
            self.writes.append(restitution.clone())

    h = _Bouncy()
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        h.set_body_restitution("leg", torch.full((2, 3), 1.5))
    h.set_body_restitution("leg", torch.full((2, 3), 0.5))
    assert h.writes[-1].tolist() == [[0.5] * 3] * 2


def test_an_unimplemented_hook_raises_even_for_friction():
    class _Bare(_Handler):
        _get_body_friction = BaseSimHandler._get_body_friction

    with pytest.raises(NotImplementedError, match="_Bare does not expose body friction"):
        _Bare().get_body_friction("leg")
