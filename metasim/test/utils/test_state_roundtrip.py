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


@pytest.mark.general
def test_list_state_to_tensor_handles_rgb_only_camera():
    """Regression: the camera read path indexed 'rgb' AND 'depth' unconditionally.

    The write path only emits a key when non-None, so a depth-only / rgb-only
    camera dict KeyError'd. The read side now indexes each only when present.
    """
    env_state = {
        "objects": {},
        "robots": {},
        "cameras": {"cam0": {"rgb": torch.zeros(4, 4, 3)}},  # no "depth" key
    }
    ts = list_state_to_tensor(_NameStub(), [env_state])  # must not raise
    assert ts.cameras["cam0"].rgb is not None
    assert ts.cameras["cam0"].rgb.shape == (1, 4, 4, 3)
    assert ts.cameras["cam0"].depth is None


def _obj(val):
    from metasim.types import ObjectState

    return ObjectState(root_state=torch.full((1, 13), float(val)))


@pytest.mark.general
def test_join_tensor_states_concats_tensor_extras():
    """Regression: join_tensor_states dropped extras -> parallel get_states().extras empty."""
    from metasim.types import TensorState
    from metasim.utils.state import join_tensor_states

    s0 = TensorState(objects={"cube": _obj(0)}, robots={}, cameras={}, extras={"k": torch.zeros(1, 3)})
    s1 = TensorState(objects={"cube": _obj(1)}, robots={}, cameras={}, extras={"k": torch.ones(1, 3)})
    out = join_tensor_states([s0, s1])
    assert out.objects["cube"].root_state.shape[0] == 2
    assert out.extras["k"].shape == (2, 3)  # concatenated along the env axis
    assert torch.equal(out.extras["k"][0], torch.zeros(3))
    assert torch.equal(out.extras["k"][1], torch.ones(3))


@pytest.mark.general
def test_join_tensor_states_drops_partial_and_nontensor_extras():
    """A key missing from a worker, or a non-tensor extra, must be dropped (not crash)."""
    from metasim.types import TensorState
    from metasim.utils.state import join_tensor_states

    s0 = TensorState(
        objects={"cube": _obj(0)},
        robots={},
        cameras={},
        extras={"present": torch.zeros(1, 2), "partial": torch.zeros(1, 2), "nested": {"a": torch.zeros(1)}},
    )
    s1 = TensorState(
        objects={"cube": _obj(1)},
        robots={},
        cameras={},
        extras={"present": torch.ones(1, 2), "nested": {"a": torch.ones(1)}},
    )
    out = join_tensor_states([s0, s1])
    assert out.extras["present"].shape == (2, 2)
    assert "partial" not in out.extras  # missing in s1 -> dropped (length would mismatch num_envs)
    assert "nested" not in out.extras  # non-tensor -> dropped (matches dict round-trip behaviour)


@pytest.mark.general
def test_join_tensor_states_no_extras_stays_empty():
    from metasim.types import TensorState
    from metasim.utils.state import join_tensor_states

    out = join_tensor_states([
        TensorState(objects={"cube": _obj(0)}, robots={}, cameras={}),
        TensorState(objects={"cube": _obj(1)}, robots={}, cameras={}),
    ])
    assert out.extras == {}
    assert out.objects["cube"].root_state.shape[0] == 2
