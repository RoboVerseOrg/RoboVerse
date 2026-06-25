"""``set_states`` normalises caller input to each backend's declared shape.

Per-backend ``_set_states`` used to disagree on the accepted state type:
Mujoco/IsaacSim/Blender took either shape, while Genesis/PyBullet/Sapien3
only took ``list[DictEnvState]`` and raised on a ``TensorState``. That
made ``handler.set_states(tensor_state)`` work or crash depending on the
active backend.

The base ``set_states`` now converts the input to the shape declared by
``_set_states_input_type`` before delegating: ``"both"`` passes through,
``"dict"`` converts a ``TensorState`` to the list form.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from metasim.sim.base import BaseSimHandler
from metasim.types import TensorState


class _Recorder(BaseSimHandler):
    """Concrete handler that records every ``_set_states`` invocation."""

    received: list = []

    def _set_states(self, states, env_ids=None):
        type(self).received.append((states, env_ids))

    def _set_dof_targets(self, actions):
        return None

    def _get_states(self, env_ids=None):
        return {}

    def _simulate(self):
        return None


def _make_handler(input_type: str) -> _Recorder:
    cls = type(
        f"_Recorder_{input_type}",
        (_Recorder,),
        {"_set_states_input_type": input_type},
    )
    h = cls.__new__(cls)
    h._tensor_state_cache = None
    h._dict_state_cache = None
    h._defer_all_visual_flushes = False
    cls.received = []
    return h


def _empty_tensor_state() -> TensorState:
    return TensorState(objects={}, robots={}, cameras={})


@pytest.fixture(autouse=True)
def _reset_recorder():
    _Recorder.received = []


@pytest.mark.general
def test_default_input_type_is_both():
    """Handlers that declare nothing keep the no-conversion behaviour."""
    assert BaseSimHandler._set_states_input_type == "both"


@pytest.mark.general
def test_both_handler_receives_input_unchanged():
    h = _make_handler("both")
    payload = {"sentinel": True}
    h.set_states(payload)
    assert type(h).received == [(payload, None)]


@pytest.mark.general
def test_dict_handler_converts_tensor_state_to_list_form():
    """A ``"dict"`` handler must never see a raw ``TensorState``."""
    h = _make_handler("dict")
    ts = _empty_tensor_state()

    with patch("metasim.sim.base.state_tensor_to_nested") as mock_convert:
        mock_convert.return_value = [SimpleNamespace(marker="converted-dict")]
        h.set_states(ts)
        mock_convert.assert_called_once_with(h, ts)

    assert len(type(h).received) == 1
    received_states, _ = type(h).received[0]
    assert received_states[0].marker == "converted-dict"


@pytest.mark.general
def test_dict_handler_passes_through_list_input_unchanged():
    h = _make_handler("dict")
    payload = [{"objects": {}, "robots": {}}]
    h.set_states(payload)
    assert type(h).received == [(payload, None)]


@pytest.mark.general
def test_env_ids_pass_through_unchanged():
    """The ``env_ids`` argument must not be touched by the normaliser."""
    h = _make_handler("both")
    h.set_states({"x": 1}, env_ids=[0, 2])
    assert type(h).received == [({"x": 1}, [0, 2])]


@pytest.mark.general
def test_parallel_wrapper_declares_dict_input_type():
    """``ParallelSimWrapper`` workers receive a per-env list of dicts over the
    pipe, and the wrapper's ``_set_states`` indexes/``len()``s that list. So
    the generated ``ParallelHandler`` must declare ``"dict"`` — otherwise a
    ``TensorState`` (e.g. from ``RLTaskEnv``'s ``list_state_to_tensor``) reaches
    ``_set_states`` unconverted and crashes with
    ``TypeError: object of type 'TensorState' has no len()`` at num_envs>1.

    Checked at the class level (no worker processes spawned): the closure
    class object carries the attribute regardless of instantiation.
    """
    from metasim.sim.parallel import ParallelSimWrapper

    wrapped = ParallelSimWrapper(_Recorder)
    assert wrapped._set_states_input_type == "dict", (
        "ParallelHandler must consume the dict batch form so the base "
        "normaliser converts TensorState before _set_states"
    )


@pytest.mark.general
def test_known_dict_only_backends_declare_their_input_type():
    """Genesis / PyBullet / Sapien3 must declare ``"dict"`` so the base
    converts ``TensorState`` input for them."""
    import importlib

    expected = {
        "metasim.sim.genesis.genesis": "GenesisHandler",
        "metasim.sim.pybullet.pybullet": "SinglePybulletHandler",
        "metasim.sim.sapien.sapien3": "Sapien3Handler",
    }

    for module_path, cls_name in expected.items():
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip(f"{module_path} backend not installed in this env")
        cls = getattr(module, cls_name)
        assert cls._set_states_input_type == "dict", (
            f"{cls_name} should declare _set_states_input_type='dict'"
        )
