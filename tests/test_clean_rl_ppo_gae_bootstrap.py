"""Regression: PPO must not bootstrap across an episode boundary.

CleanRL's PPO stores ``dones[step]`` *before* stepping, so ``dones[t]`` answers "is the
state in ``obs[t]`` terminal//post-reset?". GAE relies on exactly that alignment: at step
``t`` it masks the bootstrap with ``nextnonterminal = ~dones[t + 1]``, i.e. "did the action
taken at ``t`` end the episode?".

The rollout loop used to *also* assign ``dones[step] = next_done`` after ``envs.step()``,
overwriting the pre-step value with the post-step one. That shifts the whole ``dones``
buffer one slot earlier, so on the transition that actually terminated an episode the mask
read ``dones[t + 1]`` — the flag for the *next* episode's first step, which is 0. The
terminal delta therefore became

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

where ``s_{t+1}`` is the **reset state of a fresh episode**. The agent collected a free
``gamma * V(reset)`` every time it ended an episode, which rewards terminating early —
catastrophic on tasks whose failure terminates (falling, dropping, timing out).

These tests pin the source contract (``ppo.py`` runs only under ``__main__`` and needs the
full RL stack, so it is not importable here) and demonstrate the numerical consequence.
"""

from __future__ import annotations

import pathlib

import pytest

_PPO = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "rl" / "clean_rl" / "ppo.py"


@pytest.mark.general
def test_dones_is_recorded_before_stepping_only_once():
    """``dones[step]`` must be written once, before ``envs.step()`` — never after."""
    src = _PPO.read_text()
    lines = [ln.strip() for ln in src.splitlines()]

    assigns = [i for i, ln in enumerate(lines) if ln == "dones[step] = next_done"]
    assert len(assigns) == 1, (
        "ppo.py: `dones[step] = next_done` must appear exactly once (before envs.step()). "
        "A second assignment after the step overwrites the pre-step flag and shifts the "
        f"dones buffer, making GAE bootstrap across episode boundaries. Found {len(assigns)}."
    )

    step_call = next(i for i, ln in enumerate(lines) if ln.startswith("next_obs, reward, terminations"))
    assert assigns[0] < step_call, (
        "ppo.py: dones[step] is assigned *after* envs.step(); it must be recorded before, "
        "so that dones[t] describes the state in obs[t] (CleanRL's GAE alignment)."
    )

    # The GAE mask must still read the *following* step's flag.
    assert "nextnonterminal = (~dones[t + 1]).float()" in src


def _gae(rewards, values, dones, next_value, next_done, gamma=0.99, lam=0.95):
    """CleanRL's GAE, verbatim in structure, on plain floats."""
    n = len(rewards)
    adv = [0.0] * n
    lastgaelam = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            nextnonterminal = 0.0 if next_done else 1.0
            nextvalues = next_value
        else:
            nextnonterminal = 0.0 if dones[t + 1] else 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return adv


@pytest.mark.general
def test_buggy_dones_bootstrap_across_the_episode_boundary():
    """The overwritten buffer credits the terminal step with the next episode's value.

    Rollout: 4 steps, num_envs=1. The action at t=1 terminates the episode (a *failure*:
    reward 0). t=2 is the fresh episode's first state, which the critic likes (V=10).
    """
    rewards = [1.0, 0.0, 1.0, 1.0]
    values = [5.0, 5.0, 10.0, 10.0]
    terminated_at = [False, True, False, False]  # the action at t=1 ended the episode

    # Correct (pre-step) buffer: dones[t] == "obs[t] is a post-reset state".
    # The episode ended on the action at t=1, so the state at t=2 is the fresh one.
    fixed_dones = [False, False, True, False]
    # Buggy (post-step) buffer: dones[t] == "the action at t ended the episode".
    buggy_dones = terminated_at

    fixed = _gae(rewards, values, fixed_dones, next_value=10.0, next_done=False)
    buggy = _gae(rewards, values, buggy_dones, next_value=10.0, next_done=False)

    # At the terminal transition (t=1) the correct delta must NOT bootstrap: the episode
    # is over, so the target is just the reward.
    assert fixed[1] == pytest.approx(0.0 + 0.0 - 5.0)  # r=0, no bootstrap, -V(s_1)

    # The buggy version bootstraps V(s_2)=10 — the *next episode's* reset state — turning
    # a -5 advantage into a positive one. Ending the episode looks good.
    assert buggy[1] == pytest.approx(0.0 + 0.99 * 10.0 - 5.0 + 0.99 * 0.95 * buggy[2])
    assert buggy[1] > 0.0 > fixed[1], (
        "the buggy mask must make terminating look advantageous — that is the bug this test guards against"
    )
