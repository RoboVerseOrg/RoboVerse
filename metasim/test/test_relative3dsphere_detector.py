"""Regression test for ``Relative3DSphereDetector``.

Background: the detector's ``reset`` and ``is_detected`` used to call
``handler.get_pos(...)`` as if it were a handler method, but no handler
in the repo implements one — the project-standard pattern is the
module-level ``get_pos`` helper in
``metasim.example.example_pack.tasks.checkers.util`` (which every other
detector in ``detectors.py`` already imports + uses). Result: the
detector crashed with ``AttributeError`` the moment a task tried to use
it. Surfaced while porting ManiSkill PickCube-v1 success semantics.

This test uses a stub handler that mirrors the same protocol the real
detectors expect (``get_states(mode="tensor")`` returning a
``TensorState``-like object with one named object root_state field).
It exercises both ``reset()`` and ``is_detected()`` without any real
sim, so it's safe to run anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.general


def _make_handler(obj_pos_t0: torch.Tensor, obj_pos_now: torch.Tensor):
    """Stub a handler whose `get_states(mode='tensor')` returns a
    TensorState-shape with `objects["cube"].root_state` set to the given
    position. Switches between the two positions across calls so we can
    move the cube without spinning up a real sim.
    """
    state = {"step": 0}

    class _Obj:
        def __init__(self, p):
            self.root_state = p.unsqueeze(0) if p.ndim == 1 else p
            # pad to (n_env, 13): pos(3)+quat(4)+lin(3)+ang(3)
            if self.root_state.shape[-1] < 13:
                pad = torch.zeros((self.root_state.shape[0], 13 - self.root_state.shape[-1]))
                pad[:, 0] = 1.0  # quat w
                self.root_state = torch.cat([self.root_state, pad], dim=-1)

    def _get_states(env_ids=None, mode="tensor"):
        pos = obj_pos_t0 if state["step"] == 0 else obj_pos_now
        state["step"] += 1
        objects = {"cube": _Obj(pos), "goal_site": _Obj(obj_pos_t0)}
        return SimpleNamespace(objects=objects)

    return SimpleNamespace(num_envs=1, device=torch.device("cpu"), get_states=_get_states)


def test_relative3dsphere_uses_module_get_pos_and_detects():
    """reset() then is_detected() must both succeed (no AttributeError)
    and the detector must return True only when the object is inside the
    sphere."""
    from metasim.example.example_pack.tasks.checkers.detectors import Relative3DSphereDetector

    initial_pos = torch.tensor([0.5, 0.0, 0.1])
    moved_pos = torch.tensor([0.51, 0.0, 0.105])  # within 2.5 cm radius

    handler = _make_handler(initial_pos, moved_pos)
    detector = Relative3DSphereDetector(
        base_obj_name="goal_site",
        relative_pos=(0.0, 0.0, 0.0),
        radius=0.025,
    )
    detector.reset(handler)  # must not raise

    # First check on the moved-to position (should be inside the sphere).
    result = detector.is_detected(handler, "cube")
    assert result.dtype == torch.bool
    assert bool(result[0].item()) is True

    # Now move the cube far away — should be False.
    far_handler = _make_handler(initial_pos, torch.tensor([2.0, 0.0, 0.0]))
    detector_far = Relative3DSphereDetector(
        base_obj_name="goal_site",
        relative_pos=(0.0, 0.0, 0.0),
        radius=0.025,
    )
    detector_far.reset(far_handler)
    assert bool(detector_far.is_detected(far_handler, "cube")[0].item()) is False
