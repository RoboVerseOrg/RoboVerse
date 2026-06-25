"""Regression: ContactData returns a per-env (num_envs,) tensor on raw MuJoCo.

The raw-MuJoCo branch built ``has_contact = np.zeros(num_envs)`` but, on a
detected contact, rebound the name to a bare ``True`` — so ``torch.tensor(True)``
returned a 0-d scalar, while the MJX branch returns shape ``(num_env,)``. Same
query, divergent output shape across backends. This pins the (num_envs,) shape
in both the contact-present and contact-absent cases, with a stub handler (no
mujoco/jax/GPU needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _mujoco_stub(contacts):
    """A handler whose module looks like the raw-MuJoCo backend."""
    stub_cls = type("StubMujocoHandler", (), {})
    stub_cls.__module__ = "metasim.sim.mujoco.stub"  # selects the raw-MuJoCo branch
    h = stub_cls()
    h.num_envs = 1
    h.device = "cpu"
    h.data = SimpleNamespace(ncon=len(contacts), contact=contacts)
    return h


def _query(handler):
    from roboverse_pack.queries.contact import ContactData

    q = ContactData("geom_a", "geom_b")
    q._geom1_id = 10
    q._geom2_id = 20
    q.handler = handler  # bypass bind_handler (which needs a real mj model)
    return q


@pytest.mark.general
def test_contact_present_returns_per_env_tensor():
    contacts = [SimpleNamespace(geom1=10, geom2=20, dist=0.0)]  # matching pair, in contact
    out = _query(_mujoco_stub(contacts))()
    assert out.shape == (1,), f"expected (num_envs,) = (1,), got {tuple(out.shape)}"
    assert out.dtype == torch.float32
    assert out[0] == 1.0


@pytest.mark.general
def test_contact_absent_returns_per_env_tensor():
    out = _query(_mujoco_stub([]))()  # no contacts
    assert out.shape == (1,)
    assert out[0] == 0.0
