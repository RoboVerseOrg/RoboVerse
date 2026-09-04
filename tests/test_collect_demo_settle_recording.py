"""Settling one reset env steps the whole batch; those steps must land in the other in-flight demos.

With ``num_envs > 1``, ``force_reset_to_state`` settles the reset env with ``ensure_clean_state``, which
calls ``handler.simulate()`` on every env. Before the fix the other envs' demos silently skipped that
physics (a velocity jump with no action change), so their ``episode.npz`` did not replay.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from roboverse_learn.il.utils.clean_state import ensure_clean_state


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
