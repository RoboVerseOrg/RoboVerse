"""``ensure_clean_state`` settles a scene and checks the reset against the demo's initial state."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from loguru import logger as log

from roboverse_learn.il.utils.clean_state import ensure_clean_state


class _Stub:
    """A handler whose drawer keeps moving for ``moving_steps`` steps, then holds still; the arm always drifts."""

    def __init__(self, moving_steps: int, final_q: float = 0.5):
        self.moving_steps = moving_steps
        self.final_q = final_q
        self.steps = 0

    def simulate(self):
        self.steps += 1

    def get_joint_names(self, name, sort=True):
        return ["joint_a", "joint_b"]

    def get_states(self, mode="tensor"):
        moving = self.steps < self.moving_steps
        q = torch.tensor([[self.final_q + (0.01 * self.steps if moving else 0.0), 0.2]])
        pos = torch.tensor([[0.0, 0.0, 1.0 - (0.05 * self.steps if moving else 0.0)] + [0.0] * 10])
        arm = torch.tensor([[0.1 * self.steps] * 2])  # sags every step: must not block settling
        return SimpleNamespace(
            objects={"drawer": SimpleNamespace(joint_pos=q, root_state=pos)},
            robots={"franka": SimpleNamespace(joint_pos=arm, root_state=torch.zeros(1, 13))},
        )


def test_settles_after_two_quiet_steps_and_stops_early():
    h = _Stub(moving_steps=3)
    assert ensure_clean_state(h) is True
    assert h.steps == 5  # steps 1-3 move, steps 4 and 5 are the two quiet ones


def test_keeps_stepping_while_objects_move_up_to_max_steps_and_warns():
    messages = []
    sink = log.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        h = _Stub(moving_steps=100)
        assert ensure_clean_state(h, max_steps=10) is False
        assert h.steps == 10
        assert any("did not settle" in m for m in messages)
    finally:
        log.remove(sink)


def test_demo_dict_expected_state_is_checked_and_mismatch_is_reported():
    messages = []
    sink = log.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        ok = ensure_clean_state(
            _Stub(2), {"objects": {"drawer": {"dof_pos": {"joint_a": 0.5, "joint_b": 0.2}}}, "robots": {}}
        )
        assert ok is True and not messages
        bad = ensure_clean_state(_Stub(2), {"objects": {"drawer": {"dof_pos": {"joint_a": 0.9}}}, "robots": {}})
        assert bad is False
        assert messages and "drawer.joint_a" in messages[-1]
    finally:
        log.remove(sink)


def test_validation_reads_the_reset_env_not_env_zero():
    """collect_demo resets one env at a time; env 1 is validated when env_id=1 is passed."""

    class _TwoEnvs(_Stub):
        def get_states(self, mode="tensor"):
            q = torch.tensor([[0.5, 0.2], [0.9, 0.2]])  # env 0 at the demo pose, env 1 elsewhere
            return SimpleNamespace(
                objects={"drawer": SimpleNamespace(joint_pos=q, root_state=torch.zeros(2, 13))}, robots={}
            )

    expected = {"objects": {"drawer": {"dof_pos": {"joint_a": 0.5, "joint_b": 0.2}}}, "robots": {}}
    assert ensure_clean_state(_TwoEnvs(0), expected, env_id=0) is True
    assert ensure_clean_state(_TwoEnvs(0), expected, env_id=1) is False


def test_robot_drift_and_robot_expectations_are_ignored():
    """The arm sags every step in the stub; settling still completes, and robot joints are not validated."""
    h = _Stub(moving_steps=0)
    assert ensure_clean_state(h, {"robots": {"franka": {"dof_pos": {"joint_a": 99.0}}}, "objects": {}}) is True
    assert h.steps == 4
