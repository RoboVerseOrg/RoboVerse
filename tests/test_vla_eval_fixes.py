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
