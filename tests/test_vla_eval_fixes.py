"""Regression: VLA eval success metric and OpenVLA num_envs guard.

- SmolVLA and pi0 eval loops set ``stats["success"] = True`` on ``terminated OR
  truncated`` — a timeout (truncated) was counted as a success, inflating the
  reported success rate. Only ``terminated`` is success (OpenVLA gets this right).
- OpenVLA's predict_action only consumes env 0's image and emits one action, so
  ``num_envs > 1`` silently produces a wrong-shaped action; the runner must fail
  fast.

The eval loops need a model + env to run, so these are source-level guards.
"""

from __future__ import annotations

import pathlib

import pytest

_VLA = pathlib.Path(__file__).resolve().parents[1] / "roboverse_learn" / "vla"


@pytest.mark.general
def test_smolvla_eval_only_terminated_is_success():
    src = (_VLA / "SmolVLA" / "smolvla_eval.py").read_text()
    assert "if is_terminated or is_truncated:" not in src, "timeout must not be counted as success"
    assert "if is_terminated:" in src
    assert "elif is_truncated:" in src


@pytest.mark.general
def test_pi0_eval_only_terminated_is_success():
    src = (_VLA / "pi0" / "pi_eval.py").read_text()
    assert "if term or trunc:" not in src, "timeout must not be counted as success"
    assert "if term:" in src
    assert "elif trunc:" in src


@pytest.mark.general
def test_openvla_eval_rejects_multi_env():
    src = (_VLA / "OpenVLA" / "vla_eval.py").read_text()
    assert "if num_envs != 1:" in src
    assert "raise ValueError" in src


@pytest.mark.general
def test_evaluators_refuse_a_zero_episode_run_instead_of_reporting_it():
    """``total_successes / total_episodes`` with ``--num_episodes 0`` was a ZeroDivisionError (SmolVLA,
    OpenVLA) or a printed 0 % (pi0); the count is validated at the CLI boundary in all three."""
    for rel in ("SmolVLA/smolvla_eval.py", "OpenVLA/vla_eval.py", "pi0/pi_eval.py"):
        src = (_VLA / rel).read_text()
        assert "if args.num_episodes < 1:" in src, rel
        assert "--num_episodes must be >= 1" in src, rel


@pytest.mark.general
def test_act_eval_runner_rejects_multi_env_and_scores_the_env_it_runs():
    """The legacy ACT evaluator drives one env (env 0's observation, one action, one video, a single-env
    ensembling buffer). It previously accepted ``--num_envs N`` and scored env 0 only while dividing by
    the episode count; now it refuses N != 1, and its loop is bounded by the buffers it allocates."""
    src = (_VLA.parent / "il" / "policies" / "act" / "act_eval_runner.py").read_text()
    assert 'raise ValueError(f"act_eval_runner evaluates one env per episode' in src
    assert "MaxStep = 800" not in src and "MaxStep = max_timesteps" in src
    assert "SuccessOnce" not in src, "the per-env bookkeeping that scored only env 0 is gone"
    assert "actions_populated" not in src, "overlapping chunks are selected by position, as in BaseEvalRunner"
