"""Regression tests for two LIBERO content bugs (backend-free).

- scene3 "turn on the stove" termination did ``(stove_joint > thr).all(dim=-1)``
  on a ``(N,)`` tensor, collapsing the per-env result to a scalar (reducing over
  the batch). The sibling scene9 does it correctly.
- two pick tasks had registration-id typos (a misspelled alias and a primary id
  missing the ``pick_`` prefix that every sibling uses).
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from metasim.types import ObjectState, TensorState

_LIBERO = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "libero"


@pytest.mark.general
def test_stove_terminated_is_per_env():
    from roboverse_pack.tasks.libero_90.libero_kitchen_scene3_turn_on_the_stove import (
        LiberoKitchenScene3TurnOnStoveTask,
    )

    states = TensorState(
        objects={
            "flat_stove": ObjectState(
                root_state=torch.zeros(2, 13),
                joint_pos=torch.tensor([[0.6], [0.1]]),  # env0 on, env1 off
                joint_vel=torch.zeros(2, 1),
            )
        },
        robots={},
        cameras={},
    )
    task = LiberoKitchenScene3TurnOnStoveTask.__new__(LiberoKitchenScene3TurnOnStoveTask)
    out = task._terminated(states)
    assert out.shape == (2,), f"_terminated must be per-env (2,), got {tuple(out.shape)}"
    assert out.tolist() == [True, False]


@pytest.mark.general
def test_libero_pick_registration_ids_are_correct():
    soup = (_LIBERO / "libero_pick_alphabet_soup.py").read_text()
    assert "pick_alphnabet_soup" not in soup, "misspelled alias 'pick_alphnabet_soup'"
    assert '"libero.pick_alphabet_soup", "pick_alphabet_soup"' in soup

    juice = (_LIBERO / "libero_pick_orange_juice.py").read_text()
    assert '"libero.orange_juice"' not in juice, "primary id is missing the 'pick_' prefix"
    assert '"libero.pick_orange_juice", "pick_orange_juice"' in juice
