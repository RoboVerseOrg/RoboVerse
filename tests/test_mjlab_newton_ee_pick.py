"""mjlab's Newton end-effector fallback picks the deepest body of the tree, not the alphabetically last.

The state names Newton bodies by their last label segment; the depth lives in the label
(``yam/worldbody/.../link_right_finger/rf_rot/rf_down``). Counting ``/`` in the state's names gave 0 for
every body, so ``max`` fell back to the last name: ``rf_rot`` instead of ``rf_down``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from roboverse_pack.tasks.mjlab.mdp.rewards import _newton_robot_ee_pos_w

_LABELS = [
    "yam/worldbody/base",
    "yam/worldbody/base/link1",
    "yam/worldbody/base/link1/link_right_finger",
    "yam/worldbody/base/link1/link_right_finger/rf_rot",
    "yam/worldbody/base/link1/link_right_finger/rf_rot/rf_down",
    "yam/worldbody/base/link1/camera_mount",
]


def _env():
    names = sorted(label.split("/")[-1] for label in _LABELS)
    body_state = torch.zeros(1, len(names), 13)
    body_state[0, :, 0] = torch.arange(len(names), dtype=torch.float32)  # x = row index
    state = SimpleNamespace(robots={"yam": SimpleNamespace(body_names=names, body_state=body_state)})
    handler = SimpleNamespace(
        get_states=lambda mode="tensor": state,
        _model=SimpleNamespace(body_key=list(_LABELS)),
        _get_body_indices=lambda env_id, name: list(range(len(_LABELS))),
    )
    return SimpleNamespace(handler=handler, scenario=SimpleNamespace(robots=[SimpleNamespace(name="yam")])), names


def test_the_fallback_is_the_deepest_body():
    env, names = _env()
    pos = _newton_robot_ee_pos_w(env)
    assert names[int(pos[0, 0])] == "rf_down"


def test_a_named_suffix_still_wins():
    env, names = _env()
    assert names[int(_newton_robot_ee_pos_w(env, body_suffix="link1")[0, 0])] == "link1"
