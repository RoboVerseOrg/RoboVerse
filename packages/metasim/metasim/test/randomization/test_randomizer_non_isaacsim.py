"""Contract test: BaseRandomizerType binds cleanly on non-IsaacSim handlers.

Before the H4 fix, ``BaseRandomizerType.bind_handler`` unconditionally called
``IsaacSimAdapter(handler)`` whose ``__init__`` does ``import omni``. In any
env without Omniverse installed (mujoco, pybullet, sapien, genesis, blender)
the entire randomization subsystem crashed at the first bind, with no way
to opt out.

After the fix the adapter is only built when the handler is an IsaacSim /
IsaacLab handler (duck-typed by class name to avoid the IsaacSim handler
module's own omni import). Non-IsaacSim handlers get ``adapter = None``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.randomization.base import BaseRandomizerType


class _FakeMujocoHandler:
    """Duck-typed handler that quacks like the real MujocoHandler.

    Importantly: not subclassed from IsaacsimHandler / IsaaclabHandler, and
    its module path doesn't pull in Omniverse libraries.
    """

    def __init__(self) -> None:
        self.scenario = SimpleNamespace(robots=[], objects=[], cameras=[], scene=None, ground=None)

    def get_extra(self):
        return {}


class _SimpleRandomizer(BaseRandomizerType):
    REQUIRES_HANDLER = "render"

    def __call__(self, *args, **kwargs):
        return None


@pytest.mark.general
def test_base_randomizer_binds_on_non_isaacsim_handler_without_omni():
    """Binding any randomizer to a non-IsaacSim handler must not raise
    ImportError on omni / pxr / Omniverse modules."""
    handler = _FakeMujocoHandler()
    randomizer = _SimpleRandomizer()

    # The bind should complete without an exception even though omni is
    # absent. ``adapter`` ends up None — randomizers that need USD ops
    # should guard with `if self.adapter is None: skip` at call time.
    try:
        randomizer.bind_handler(handler)
    except ImportError as exc:
        pytest.fail(f"Non-IsaacSim handler bind unexpectedly imported Omniverse: {exc}")

    assert randomizer.adapter is None, "adapter should be None for non-Isaac handlers"
    assert randomizer.handler is handler


@pytest.mark.general
def test_base_randomizer_skips_isaacsim_adapter_for_blender_class_name():
    """Class-name detection: anything that's not IsaacsimHandler / IsaaclabHandler
    is treated as non-Isaac. Blender / sapien / etc. quack into this branch."""

    class _BlenderHandler:
        scenario = SimpleNamespace(robots=[], objects=[], cameras=[], scene=None, ground=None)

        def get_extra(self):
            return {}

    randomizer = _SimpleRandomizer()
    randomizer.bind_handler(_BlenderHandler())
    assert randomizer.adapter is None


@pytest.mark.general
def test_base_randomizer_attempts_isaacsim_adapter_when_handler_named_isaacsim(monkeypatch):
    """When the handler's class name matches an IsaacSim variant, the
    adapter is attempted. We patch IsaacSimAdapter to a no-op so the test
    doesn't actually require Omniverse."""
    constructed: list = []

    class _FakeAdapter:
        def __init__(self, h):
            constructed.append(h)
            self.handler = h

    import metasim.randomization.base as base_mod

    monkeypatch.setattr(base_mod, "IsaacSimAdapter", _FakeAdapter)

    class IsaacsimHandler:  # name matters for the duck-typed check
        scenario = SimpleNamespace(robots=[], objects=[], cameras=[], scene=None, ground=None)

        def get_extra(self):
            return {}

    randomizer = _SimpleRandomizer()
    randomizer.bind_handler(IsaacsimHandler())
    assert len(constructed) == 1, "should construct exactly one adapter for IsaacsimHandler"
    assert isinstance(randomizer.adapter, _FakeAdapter)
