"""Regression: TD3/SAC delayed policy update must be gated on a per-iteration counter.

``global_step`` advances by ``num_envs`` every loop, so ``global_step % policy_frequency``
is constant (always 0 whenever ``policy_frequency`` divides ``num_envs`` — true for the
defaults num_envs=128, policy_frequency=2). The "delayed" TD3/SAC update therefore ran
*every* iteration, defeating the mechanism; in SAC it compounded with the inner
``for _ in range(policy_frequency)`` loop, over-training the actor by ``policy_frequency``x.

The fix gates the actor/target updates on ``update_count`` (incremented once per gradient
step) instead of ``global_step``. These tests pin that source contract (the training
scripts run only under ``__main__`` and need the full RL stack, so they aren't importable
here) and document the intended cadence arithmetic.
"""

from __future__ import annotations

import pathlib

import pytest

_RL = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "rl" / "clean_rl"


def _src(name: str) -> str:
    return (_RL / name).read_text()


@pytest.mark.general
@pytest.mark.parametrize("fname", ["td3.py", "sac.py"])
def test_delayed_update_gate_uses_update_count_not_global_step(fname):
    src = _src(fname)
    # The buggy gate must be gone, and the counter-based gate present.
    assert "global_step % args.policy_frequency" not in src, (
        f"{fname}: delayed policy update is gated on global_step (advances by num_envs) — "
        f"it never delays. Gate on update_count instead."
    )
    assert "update_count % args.policy_frequency" in src, f"{fname}: policy update must gate on update_count"
    # The counter must be initialised and incremented once per gradient step.
    assert "update_count = 0" in src, f"{fname}: update_count is not initialised"
    assert "update_count += 1" in src, f"{fname}: update_count is never incremented"
    # The per-100 logging cadence intentionally stays on global_step (env-step semantics).
    assert "global_step % 100 == 0" in src
    # Because the actor update is now genuinely delayed (decoupled from the %100
    # logging), actor_loss can be unset on a logging iteration — it must be seeded
    # to None and the log guarded, or training crashes (UnboundLocalError) for
    # num_envs that are multiples of 100.
    assert "actor_loss = None" in src, f"{fname}: actor_loss must be seeded before the loop"
    assert "if actor_loss is not None:" in src, f"{fname}: the actor_loss log must be guarded"


@pytest.mark.general
def test_sac_target_update_also_uses_update_count():
    """SAC's target-network update shares the same fragility and must use the counter."""
    src = _src("sac.py")
    assert "global_step % args.target_network_frequency" not in src
    assert "update_count % args.target_network_frequency" in src


def _fired_iterations(num_envs, policy_frequency, n_iters, learning_starts=10, use_counter=True):
    """Iterations on which the delayed update fires, for the buggy vs fixed gate."""
    global_step = 0
    update_count = 0
    fired = []
    for it in range(n_iters):
        global_step += num_envs
        if global_step > learning_starts:
            update_count += 1
            gate = update_count if use_counter else global_step
            if gate % policy_frequency == 0:
                fired.append(it)
    return fired


@pytest.mark.general
def test_cadence_arithmetic_buggy_vs_fixed():
    """Document the behaviour: buggy gate fires every iteration; fixed fires every Nth."""
    # Buggy: global_step (multiple of 128) % 2 == 0 always -> fires every post-warmup iter.
    buggy = _fired_iterations(num_envs=128, policy_frequency=2, n_iters=20, use_counter=False)
    assert len(buggy) == 20
    # Fixed: fires exactly every policy_frequency-th gradient step.
    fixed = _fired_iterations(num_envs=128, policy_frequency=2, n_iters=20, use_counter=True)
    assert fixed == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    # Fixed cadence is independent of num_envs once past warmup (learning_starts=0
    # so update_count increments from iter 0 in both): every policy_frequency-th step.
    a = _fired_iterations(1, 3, 30, learning_starts=0, use_counter=True)
    b = _fired_iterations(997, 3, 30, learning_starts=0, use_counter=True)
    assert a == b == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
