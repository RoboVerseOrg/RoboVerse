"""Contract tests for the parity harnesses in ``scripts/``.

These scripts are the *evidence base* for RoboVerse's central claim -- that a task
behaves identically across backends / through a passthrough. AGENTS.md: "A task that
'matches' only because both sides are equally broken ... is not parity."

A harness that cannot evaluate a side must therefore **error**, never quietly score a
match. The three false-PASS shapes pinned here:

1. an exception on *both* sides collapsing to ``False == False`` -> "succ_match";
2. an observation comparison over the *intersection* of key sets, so an env that
   returns ``{}`` scores a perfect bitwise obs parity over zero keys;
3. a ``main()`` whose exit status does not reflect the verdict, so a failing parity
   run is indistinguishable from a passing one to CI.

Everything here drives pure comparison / verdict code with fakes -- no simulator, no
GPU -- so it runs in the default test environment.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import scripts.eval_liberoplus_policy_consistency as evalc
import scripts.parity_liberoplus_passthrough as parity_pt
import scripts.parity_simpler_env as parity_se
import scripts.spike_metasim_full_parity as spike


class _Boom(RuntimeError):
    """Stands in for "upstream renamed/moved ``_check_success``"."""


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class _FakeSim:
    def get_state(self):
        class _S:
            time = 0.0
            qpos = np.zeros(3)
            qvel = np.zeros(3)

        return _S()


class _FakeInner:
    def __init__(self, *, success_raises: bool):
        self.sim = _FakeSim()
        self._success_raises = success_raises

    def _check_success(self):
        if self._success_raises:
            raise _Boom("upstream renamed _check_success")
        return False


class _FakeEnv:
    """A LIBERO-ish env: identical on both sides, so only the harness logic decides."""

    def __init__(self, *, obs: dict, success_raises: bool = False):
        self.env = _FakeInner(success_raises=success_raises)
        self._obs = obs

    def set_init_state(self, state):
        return self._obs

    def step(self, action):
        return dict(self._obs), 0.0, False, {}

    def close(self):
        pass


def _patch_liberoplus_eval(monkeypatch, *, obs: dict, success_raises: bool = False) -> None:
    """Point the eval harness at two identical fake envs (no sim, no demo file)."""
    monkeypatch.setattr(evalc, "CASES", [("libero_object", "stem", "task", "Light Conditions")])
    monkeypatch.setattr(evalc, "_load_demo", lambda suite, stem, demo_idx=0: (np.zeros((3, 7)), np.zeros(4)))
    monkeypatch.setattr(evalc, "_task_id", lambda suite, name: 0)
    monkeypatch.setattr(
        evalc.pt, "make_liberoplus_env", lambda *a, **k: _FakeEnv(obs=obs, success_raises=success_raises)
    )
    monkeypatch.setattr(evalc, "_native_env", lambda *a, **k: _FakeEnv(obs=obs, success_raises=success_raises))


# --------------------------------------------------------------------------------------
# 1. an error on both sides is an ERROR, never a match
# --------------------------------------------------------------------------------------


def test_success_check_error_is_not_a_false_match():
    """`_check_success` raising must propagate, not be swallowed into ``False``.

    ``except Exception: return False`` on both sides makes ``False == False`` read as
    perfect success parity -- two equally-broken sides scoring a PASS.
    """
    env = _FakeEnv(obs={"q": np.zeros(3)}, success_raises=True)
    with pytest.raises(RuntimeError):
        evalc._success(env)


def test_success_check_missing_is_an_error():
    """No success checker at all means the harness cannot evaluate that side."""

    class _NoChecker:
        env = object()

    with pytest.raises(RuntimeError):
        evalc._success(_NoChecker())


def test_liberoplus_eval_does_not_pass_when_both_sides_raise(monkeypatch, capsys):
    """Both sides failing the success check must not print PASS / exit 0."""
    _patch_liberoplus_eval(monkeypatch, obs={"q": np.zeros(3)}, success_raises=True)
    with pytest.raises(RuntimeError):
        evalc.run(max_steps=2)
    assert "RESULT: PASS" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# 2. an empty / mismatched obs dict must not score perfect parity
# --------------------------------------------------------------------------------------


def test_liberoplus_eval_rejects_empty_observations(monkeypatch, capsys):
    """An env returning ``{}`` compares zero keys -- that is not obs parity."""
    _patch_liberoplus_eval(monkeypatch, obs={})
    with pytest.raises(RuntimeError):
        evalc.run(max_steps=2)
    assert "RESULT: PASS" not in capsys.readouterr().out


def test_obs_diff_requires_equal_key_sets():
    """A key present on one side only is a failure, not an unexamined key."""
    a = [{"joint_states": np.zeros(3), "object_states": np.zeros(2)}]
    b = [{"joint_states": np.zeros(3)}]
    with pytest.raises(RuntimeError):
        evalc._obs_diff(a, b)


def test_obs_diff_still_measures_a_real_delta():
    """The happy path must keep working: equal key sets -> the real max|Δ|, images skipped."""
    a = [{"q": np.zeros(3), "agentview_image": np.zeros((2, 2))}]
    b = [{"q": np.array([0.0, 0.25, 0.0]), "agentview_image": np.ones((2, 2))}]
    assert evalc._obs_diff(a, b) == pytest.approx(0.25)


def test_passthrough_compare_rejects_empty_obs():
    """`_compare` over ``{}`` vs a real obs dict must error, not report Δ=0."""
    traj_a = [{"obs": {"joint_states": np.zeros(3)}, "rew": 0.0, "done": False}]
    traj_b = [{"obs": {}, "rew": 0.0, "done": False}]
    with pytest.raises(RuntimeError):
        parity_pt._compare(traj_a, traj_b)


def test_passthrough_compare_rejects_two_empty_obs():
    """Two empty obs dicts agree on nothing; scoring them Δ=0 is a false PASS."""
    traj = [{"obs": {}, "rew": 0.0, "done": False}]
    with pytest.raises(RuntimeError):
        parity_pt._compare(traj, [{"obs": {}, "rew": 0.0, "done": False}])


def test_passthrough_compare_still_measures_a_real_delta():
    """The happy path must keep working: equal key sets -> the real max|Δ|."""
    traj_a = [{"obs": {"q": np.zeros(3), "agentview_image": np.zeros((2, 2))}, "rew": 1.0, "done": False}]
    traj_b = [{"obs": {"q": np.array([0.0, 0.5, 0.0]), "agentview_image": np.ones((2, 2))}, "rew": 1.0, "done": False}]
    dev_state, dev_img, dev_rew, done_match = parity_pt._compare(traj_a, traj_b)
    assert dev_state == pytest.approx(0.5)
    assert dev_img == pytest.approx(1.0)
    assert dev_rew == 0.0
    assert done_match


def test_passthrough_run_does_not_vacuously_pass(monkeypatch, capsys):
    """Zero sampled tasks verifies nothing; ``0/0`` must not be a PASS."""
    monkeypatch.setattr(parity_pt, "_sample_tasks", lambda per_dim=1: [])
    with pytest.raises(RuntimeError):
        parity_pt.run(per_dim=1, steps=2, seed=0)
    assert "RESULT: PASS" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# 3. exit status must reflect the verdict
# --------------------------------------------------------------------------------------


def test_spike_full_parity_exits_nonzero_on_failure(monkeypatch):
    """A failing parity run must be visible to a shell/CI caller."""
    monkeypatch.setattr(spike, "run_compare", lambda: False)
    monkeypatch.setattr("sys.argv", ["spike_metasim_full_parity.py", "--mode", "compare"])
    with pytest.raises(SystemExit) as exc:
        spike.main()
    assert exc.value.code != 0


def test_spike_full_parity_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr(spike, "run_compare", lambda: True)
    monkeypatch.setattr("sys.argv", ["spike_metasim_full_parity.py", "--mode", "compare"])
    with pytest.raises(SystemExit) as exc:
        spike.main()
    assert exc.value.code == 0


def test_spike_full_parity_data_root_is_env_or_repo_local():
    """The asset root follows ``ROBOVERSE_DATA_DIR`` or the repo checkout — never a developer's home."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(spike.__file__)))
    expected = os.environ.get("ROBOVERSE_DATA_DIR", os.path.join(repo_root, "roboverse_data"))
    assert spike.RV == expected
    assert "/home/ghr" not in spike.RV


def test_spike_compare_rejects_truncated_rollout(monkeypatch, tmp_path):
    """Comparing only ``min(len(a), len(b))`` frames hides a side that died early."""
    monkeypatch.setattr(spike, "ART", str(tmp_path))
    monkeypatch.setattr(spike, "ALL", ["t"])
    np.savez(tmp_path / "mfp_native_t.npz", rgbs=np.zeros((4, 2, 2, 3)), succ=np.zeros(3, bool))
    np.savez(tmp_path / "mfp_metasim_t.npz", rgbs=np.zeros((1, 2, 2, 3)), succ=np.zeros(0, bool))
    assert spike.run_compare() is False


def test_simpler_env_parity_main_exits_nonzero_on_diff(monkeypatch, capsys):
    monkeypatch.setattr(parity_se, "check_task", lambda t, **kw: {"task": t, "parity_1to1": False, "diffs": [{}]})
    monkeypatch.setattr("sys.argv", ["parity_simpler_env.py", "--tasks", "some_task"])
    assert parity_se.main() != 0
    assert "RESULT: PASS" not in capsys.readouterr().out


def test_simpler_env_parity_main_exits_zero_when_all_match(monkeypatch):
    monkeypatch.setattr(parity_se, "check_task", lambda t, **kw: {"task": t, "parity_1to1": True, "diffs": []})
    monkeypatch.setattr("sys.argv", ["parity_simpler_env.py", "--tasks", "some_task"])
    assert parity_se.main() == 0


def test_simpler_env_parity_main_does_not_vacuously_pass(monkeypatch):
    """No tasks checked -> nothing proven -> not a PASS."""
    monkeypatch.setattr(parity_se, "check_task", lambda t, **kw: {"task": t, "parity_1to1": True})
    monkeypatch.setattr("sys.argv", ["parity_simpler_env.py", "--tasks"])
    assert parity_se.main() != 0
