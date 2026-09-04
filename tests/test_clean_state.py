"""``ensure_clean_state`` settles a scene and checks the reset against the demo's initial state."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from loguru import logger as log

from roboverse_learn.il.utils.clean_state import ensure_clean_state, settle_recipients


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


def test_on_step_receives_the_full_state_after_every_simulate():
    """A batched recorder keeps the settle steps of the other envs; it needs every intermediate state."""
    h = _Stub(moving_steps=2)
    seen = []
    ensure_clean_state(h, on_step=lambda state: seen.append(float(state.objects["drawer"].joint_pos[0, 0])))
    assert len(seen) == h.steps == 4


# --- settling one reset env steps the whole batch; those steps must land in the other in-flight demos


class _Batch:
    """Two-env handler whose objects come to rest after two steps; each step is recorded by the callback."""

    num_envs = 2

    def __init__(self):
        self.steps = 0

    def simulate(self):
        self.steps += 1

    def get_joint_names(self, name, sort=True):
        return ["j"]

    def get_states(self, mode="tensor"):
        moving = self.steps < 2
        t = self.steps if moving else 2  # everything comes to rest after two steps
        pos = torch.tensor([[0.0, 0.0, 1.0 - 0.1 * t] + [0.0] * 10] * 2)
        q = torch.tensor([[0.1 * t], [0.2 * t]])
        return SimpleNamespace(
            objects={"cube": SimpleNamespace(joint_pos=None, root_state=pos)},
            robots={"arm": SimpleNamespace(joint_pos=q, joint_pos_target=q.clone(), root_state=torch.zeros(2, 13))},
        )


def test_settle_steps_are_delivered_to_the_recorder_for_every_env():
    h = _Batch()
    recorded = {0: [], 1: []}

    def keep_other_envs(state):
        for env in (0, 1):
            recorded[env].append(float(state.robots["arm"].joint_pos[env, 0]))

    ensure_clean_state(h, on_step=keep_other_envs)
    assert h.steps == 4 and len(recorded[0]) == 4 and len(recorded[1]) == 4
    assert [round(v, 4) for v in recorded[1]] == [
        0.2,
        0.4,
        0.4,
        0.4,
    ]  # env 1 advanced during env 0's settle, and it was seen


def test_settle_recipients_are_the_other_envs_that_keep_recording():
    """The reset env, finished envs, envs whose demo closes this iteration and envs without an open
    demo receive nothing; a reset env that is recording again (not terminal any more) does."""
    kw = dict(env_id=1, finished=[False, False, True, False, False], terminal={3}, recording=[0, 1, 2, 3])
    assert settle_recipients(5, **kw) == [0]
    kw["terminal"] = set()  # env 3 was reset and records again
    assert settle_recipients(5, **kw) == [0, 3]
    assert settle_recipients(5, env_id=None, finished=None, terminal=None, recording=[4]) == [4]
    assert settle_recipients(2, env_id=0, recording=[]) == []
