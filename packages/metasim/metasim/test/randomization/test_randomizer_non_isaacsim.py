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


@pytest.mark.general
def test_subclass_adapter_errors_are_not_silenced(monkeypatch):
    import metasim.randomization.base as base_mod

    class IsaacsimHandler(_FakeMujocoHandler):
        pass

    class CustomHandler(IsaacsimHandler):
        pass

    def broken_adapter(handler):
        raise ImportError("missing USD dependency")

    monkeypatch.setattr(base_mod, "IsaacSimAdapter", broken_adapter)
    with pytest.raises(ImportError, match="missing USD"):
        _SimpleRandomizer().bind_handler(CustomHandler())


@pytest.mark.general
def test_registry_isolation_survives_switching_and_garbage_collection():
    import gc

    from metasim.randomization.core import ObjectMetadata, ObjectRegistry

    ObjectRegistry.reset()
    first, second = _FakeMujocoHandler(), _FakeMujocoHandler()
    registry = ObjectRegistry.get_instance(first)
    registry.register(ObjectMetadata("a", "object", "dynamic", ["/a/0", "/a/1"]))
    ObjectRegistry.get_instance(second)
    del registry
    gc.collect()
    assert ObjectRegistry.get_instance(first).get("a") is not None
    assert ObjectRegistry.get_instance(second).get("a") is None
    assert ObjectRegistry.get_instance() is ObjectRegistry.get_instance(second)
    ObjectRegistry.reset()
    assert ObjectRegistry.get_instance(first).get("a") is None
    ObjectRegistry.reset()


@pytest.mark.general
@pytest.mark.parametrize("shared", [True, False])
def test_registry_validates_env_indices_and_empty_selection(shared):
    import torch

    from metasim.randomization.core import ObjectMetadata, ObjectRegistry

    h = _FakeMujocoHandler()
    h.num_envs = 2
    registry = ObjectRegistry(h)
    registry.register(ObjectMetadata("a", "object", "static", ["/a"] if shared else ["/a/0", "/a/1"], shared=shared))
    assert registry.get_prim_paths("a", env_ids=[]) == []
    assert registry.get_prim_paths("a", env_ids=torch.tensor([], dtype=torch.long)) == []
    for ids in ([-1], [2], [True], [0.0], torch.tensor(0)):
        with pytest.raises(ValueError):
            registry.get_prim_paths("a", env_ids=ids)
    assert registry.get_prim_paths("a", env_ids=[1]) == (["/a"] if shared else ["/a/1"])


@pytest.mark.general
def test_camera_registry_uses_sensor_paths_including_mounted_parent():
    from metasim.randomization.core import ObjectRegistry

    handler = _FakeMujocoHandler()
    handler.num_envs = 2
    handler.cameras = [SimpleNamespace(name="view"), SimpleNamespace(name="wrist")]
    handler.scene = SimpleNamespace(
        sensors={
            "view": SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/view")),
            "wrist": SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/robot/hand/wrist")),
        }
    )
    registry = ObjectRegistry(handler)
    BaseRandomizerType._scan_and_register_handler_objects(handler, registry)
    assert registry.get_prim_paths("view", env_ids=[1]) == ["/World/envs/env_1/view"]
    assert registry.get_prim_paths("wrist", env_ids=[0]) == ["/World/envs/env_0/robot/hand/wrist"]
    assert not registry.get("wrist").shared
