"""``HybridSimHandler._set_states`` normalises input per wrapped handler.

The base ``set_states`` normalises a caller's input to the shape the handler
declared via ``_set_states_input_type`` *before* calling ``_set_states``. But
``HybridSimHandler`` overrides ``_set_states`` and forwards to the wrapped
physics/render handlers directly — so a wrapped ``"dict"`` backend
(genesis/sapien3/pybullet) used to receive an un-normalised ``TensorState`` it
cannot index. The fix runs each wrapped handler's own
``_normalise_set_states_input`` before forwarding.

Backend-free: stub physics/render handlers record what type they receive; the
Hybrid handler is built via ``__new__`` so no real simulator/scenario is needed.
"""

from __future__ import annotations

import pytest
import torch

from metasim.sim.base import BaseSimHandler
from metasim.sim.hybrid import HybridSimHandler
from metasim.types import RobotState, TensorState


class _RecordingHandler(BaseSimHandler):
    """Records the type its ``_set_states`` receives.

    ``_set_states_input_type`` is read off the *class* by
    ``_normalise_set_states_input`` (as real backends declare it), so input-typed
    handlers are built as dynamic subclasses via ``_recording_handler`` below.
    """

    def __init__(self):
        self.received_type = None

    def _set_states(self, states, env_ids=None):
        self.received_type = type(states).__name__

    def _set_dof_targets(self, actions):
        return None

    def _get_states(self, env_ids=None):
        return None  # forces the render handler down the dict-state else-branch

    def _simulate(self):
        return None

    def _get_joint_names(self, obj_name, sort=True):
        return ["j0"]

    def _get_body_names(self, obj_name, sort=True):
        return []


def _recording_handler(input_type):
    cls = type("_RecordingHandler_" + input_type, (_RecordingHandler,), {"_set_states_input_type": input_type})
    return cls()


def _make_hybrid(physics_input_type, render_input_type):
    h = object.__new__(HybridSimHandler)  # bypass scenario/backend init
    h.physics_handler = _recording_handler(physics_input_type)
    h.render_handler = _recording_handler(render_input_type)
    return h


def _tensor_state():
    robot = RobotState(root_state=torch.zeros(1, 13), joint_pos=torch.zeros(1, 1), joint_vel=torch.zeros(1, 1))
    return TensorState(objects={}, robots={"r0": robot}, cameras={})


@pytest.mark.general
def test_dict_physics_handler_receives_normalised_dict():
    """A ``"dict"`` physics handler must get a list-of-dicts, not a TensorState."""
    h = _make_hybrid(physics_input_type="dict", render_input_type="dict")
    h._set_states(_tensor_state())
    # _normalise_set_states_input converts TensorState -> list[DictEnvState] for "dict".
    assert h.physics_handler.received_type == "list"
    # render goes through the else-branch (physics _get_states returned None) and is
    # also a "dict" handler -> normalised to list as well.
    assert h.render_handler.received_type == "list"


@pytest.mark.general
def test_both_physics_handler_passes_tensorstate_through():
    """A ``"both"`` physics handler still receives the TensorState unchanged (no-op)."""
    h = _make_hybrid(physics_input_type="both", render_input_type="both")
    h._set_states(_tensor_state())
    assert h.physics_handler.received_type == "TensorState"
