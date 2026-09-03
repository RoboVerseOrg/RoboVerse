"""The newton compat shims forward renamed / removed attributes on both old and new newton APIs.

newton 1.5 replaced ``Model.joint_target_pos`` / ``joint_target_vel`` (and the same on ``Control``)
with ``joint_target_q`` / ``joint_target_qd`` behind a ``RemovedAttribute`` descriptor that raises on
read and write; 1.6 renamed ``Model.num_worlds`` to ``world_count`` outright. The shims are exercised
against a synthetic ``newton`` module so the test runs without the engine installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


class RemovedAttribute:
    """Mirror of ``newton._src.utils.deprecation.RemovedAttribute`` (data descriptor that raises)."""

    def __init__(self, replacement: str):
        self._message = f"removed; use {replacement} instead"

    def __get__(self, instance, owner=None):
        raise AttributeError(self._message)

    def __set__(self, instance, value):
        raise AttributeError(self._message)


def _fake_newton(*, removed: bool, renamed: bool) -> types.ModuleType:
    mod = types.ModuleType("newton")

    class Model:
        def __init__(self):
            if renamed:
                self.world_count = 4
            else:
                self.num_worlds = 4
            if removed:
                self.joint_target_q = "q"
                self.joint_target_qd = "qd"
            else:
                self.joint_target_pos = "q"
                self.joint_target_vel = "qd"

    class Control:
        def __init__(self):
            if removed:
                self.joint_target_q = "cq"
            else:
                self.joint_target_pos = "cq"

    if removed:
        Model.joint_target_pos = RemovedAttribute("joint_target_q")
        Model.joint_target_vel = RemovedAttribute("joint_target_qd")
        Control.joint_target_pos = RemovedAttribute("joint_target_q")
        Control.joint_target_vel = RemovedAttribute("joint_target_qd")

    class ModelBuilder:
        pass

    mod.Model, mod.Control, mod.ModelBuilder = Model, Control, ModelBuilder
    mod.JointType = object
    mod.sensors = types.ModuleType("newton.sensors")  # no populate_contacts: exercises the >=1.2 path
    return mod


def _load_compat(monkeypatch, fake):
    """Load ``_newton_compat.py`` against ``fake`` as the ``newton`` package.

    Loaded from its file so ``metasim.sim.newton/__init__`` (which imports the real engine) stays out
    of the picture.
    """
    for name in list(sys.modules):
        if name == "newton" or name.startswith("newton."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "newton", fake)
    monkeypatch.setitem(sys.modules, "newton.sensors", fake.sensors)
    import metasim

    path = os.path.join(os.path.dirname(metasim.__file__), "sim", "newton", "_newton_compat.py")
    spec = importlib.util.spec_from_file_location("_newton_compat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.general
def test_removed_joint_target_attributes_forward_to_new_names(monkeypatch):
    fake = _fake_newton(removed=True, renamed=True)
    compat = _load_compat(monkeypatch, fake)
    compat._install_removed_attr_aliases()
    compat._install_renamed_attr_aliases()
    model, control = fake.Model(), fake.Control()
    assert model.joint_target_pos == "q" and model.joint_target_vel == "qd"
    assert control.joint_target_pos == "cq"
    model.joint_target_pos = "new"
    assert model.joint_target_q == "new", "writes must land on the new attribute, not a dead shadow"
    assert model.num_worlds == 4


@pytest.mark.general
def test_old_newton_attributes_are_left_alone(monkeypatch):
    fake = _fake_newton(removed=False, renamed=False)
    compat = _load_compat(monkeypatch, fake)
    compat._install_removed_attr_aliases()
    compat._install_renamed_attr_aliases()
    model = fake.Model()
    assert model.joint_target_pos == "q" and model.num_worlds == 4
    model.num_worlds = 8
    assert model.num_worlds == 8
