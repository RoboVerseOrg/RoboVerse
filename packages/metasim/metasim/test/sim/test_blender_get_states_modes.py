"""BlenderHandler.get_states must accept the base default ``mode='dict'`` (H6).

The previous override hardcoded tensor-only AND raised NotImplementedError on
dict mode — so any caller using the base default
``handler.get_states()`` (with no kwargs) crashed on Blender. The base API
typed ``mode: StateMode = "dict"`` so this default-default-mismatch broke
every generic task / runner that wasn't Blender-aware.

The fix delegates to ``_get_states`` (tensor mode) and converts to dict
only when asked, returning a single-env empty dict when there's no physics
state to nestify (Blender is render-only).
"""

from __future__ import annotations

import pytest


def _load_blender_get_states():
    """Pull just ``BlenderHandler.get_states`` out of the module so the
    test doesn't need ``bpy`` to run."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "sim" / "blender" / "blender.py"
    if not src.is_file():
        pytest.skip(f"blender.py not at expected location: {src}")
    source = src.read_text()
    marker = "    def get_states(self, env_ids"
    idx = source.find(marker)
    if idx < 0:
        pytest.skip("could not locate BlenderHandler.get_states")
    end = source.find("\n    def ", idx + 1)
    body = source[idx:end] if end > 0 else source[idx:]
    wrapper = "class _BlenderLike:\n" + body
    ns: dict = {}
    exec(wrapper, ns)
    return ns["_BlenderLike"]


_BlenderLike = _load_blender_get_states()


class _StubTensorState:
    def __init__(self, objects=None, robots=None, cameras=None, extras=None):
        self.objects = objects or {}
        self.robots = robots or {}
        self.cameras = cameras or {}
        self.extras = extras


@pytest.mark.general
def test_get_states_default_mode_is_dict_and_returns_list_when_empty():
    """The base API default is mode='dict'; calling with no args must work."""
    h = _BlenderLike()
    h.num_envs = 2
    h._get_states = lambda env_ids=None: _StubTensorState()
    result = h.get_states()
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(r == {"objects": {}, "robots": {}} for r in result)


@pytest.mark.general
def test_get_states_tensor_mode_returns_tensor_state():
    h = _BlenderLike()
    h.num_envs = 1
    stub = _StubTensorState(cameras={"cam0": object()})
    h._get_states = lambda env_ids=None: stub
    result = h.get_states(mode="tensor")
    assert result is stub


@pytest.mark.general
def test_get_states_rejects_unknown_mode():
    h = _BlenderLike()
    h.num_envs = 1
    h._get_states = lambda env_ids=None: _StubTensorState()
    with pytest.raises(ValueError, match="Unknown state mode"):
        h.get_states(mode="nested")


@pytest.mark.general
def test_get_states_dict_mode_falls_back_when_no_objects_or_robots():
    """When physics state is empty, dict mode returns one empty dict per env
    instead of crashing inside state_tensor_to_nested (which assumes
    objects+robots is non-empty)."""
    h = _BlenderLike()
    h.num_envs = 3
    h._get_states = lambda env_ids=None: _StubTensorState()
    result = h.get_states(mode="dict")
    assert len(result) == 3
    for env_state in result:
        assert env_state == {"objects": {}, "robots": {}}
