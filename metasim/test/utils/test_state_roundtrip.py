"""General tests for metasim.utils.state helpers (no simulator backend)."""

from __future__ import annotations

import pytest
import torch

from metasim.utils.state import _alloc_state_tensors, list_state_to_tensor


@pytest.mark.general
def test_alloc_state_tensors_default_device_is_valid():
    """Regression: the default device was 'gpu' (not a valid torch device)."""
    root, body, jpos, jvel = _alloc_state_tensors(1, 1, 1)
    assert root.device.type == "cpu"
    assert root.shape == (1, 13)
    assert body.shape == (1, 1, 13)
    assert jpos.shape == (1, 1)


class _NameStub:
    """Minimal handler exposing only the name lookups list_state_to_tensor needs."""

    def get_body_names(self, name):
        return ["base", "door"]

    def get_joint_names(self, name):
        return ["joint0"]


def _body():
    return {
        "pos": torch.zeros(3),
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "vel": torch.zeros(3),
        "ang_vel": torch.zeros(3),
    }


@pytest.mark.general
def test_articulated_object_keeps_body_names_on_roundtrip():
    """Regression: list_state_to_tensor dropped body_names for objects (kept for robots)."""
    env_state = {
        "objects": {
            "cube": {
                "pos": torch.zeros(3),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "body": {"base": _body(), "door": _body()},
                "dof_pos": {"joint0": 0.5},
                "dof_vel": {"joint0": 0.0},
            }
        },
        "robots": {},
        "cameras": {},
    }
    ts = list_state_to_tensor(_NameStub(), [env_state])
    obj = ts.objects["cube"]
    assert obj.body_names == ["base", "door"], "object body_names must survive the round-trip"
    # body_names must label the body_state rows (len match -> ObjectState.__post_init__ validation).
    assert obj.body_state.shape[1] == len(obj.body_names)
