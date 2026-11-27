"""``metasim.bench``: the protocol rules, the result schema and the evaluator loop, without a simulator.

The evaluator is driven here by a scripted fake env whose per-step success / termination schedule is
known, in both reset styles (``auto_reset`` False like ``BaseTaskEnv``, True like ``RLTaskEnv``), so
the counting rules are pinned independently of any backend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from metasim.bench import (
    BenchmarkResult,
    EpisodeResult,
    EvalProtocol,
    HoldPolicy,
    PolicyContext,
    RandomPolicy,
    ReplayPolicy,
    TaskInstance,
    evaluate,
    hash_state,
    load_policy,
    plan_instances,
    summarize,
    validate_result,
    wilson_interval,
)
from metasim.bench.result import SCHEMA, SCHEMA_VERSION

pytestmark = pytest.mark.general


# ----------------------------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rule", "hold", "flags", "expected"),
    [
        ("first", 1, [False, True, False], True),
        ("first", 1, [False, False], False),
        ("final", 1, [True, True, False], False),
        ("final", 1, [False, True], True),
        ("held", 3, [True, True, False, True, True, True], True),
        ("held", 3, [True, True, False, True, True], False),
        ("held", 1, [False, True], True),
        ("first", 1, [], False),
    ],
)
def test_episode_success_rules(rule, hold, flags, expected):
    assert EvalProtocol(success_rule=rule, hold_steps=hold).episode_success(flags) is expected


def test_protocol_validation_and_roundtrip():
    with pytest.raises(ValueError, match="success_rule"):
        EvalProtocol(success_rule="sometimes")
    with pytest.raises(ValueError, match="hold_steps"):
        EvalProtocol(hold_steps=0)
    with pytest.raises(ValueError, match="episodes"):
        EvalProtocol(episodes=0)
    with pytest.raises(ValueError, match="disagrees"):
        EvalProtocol(episodes=3, instances=(0, 1))
    p = EvalProtocol(instances=[2, 0], seed=7, success_rule="held", hold_steps=4)
    assert p.episodes == 2 and p.instances == (2, 0)
    assert EvalProtocol.from_dict(json.loads(json.dumps(p.to_dict()))) == p


# ----------------------------------------------------------------------------------------------
# instances
# ----------------------------------------------------------------------------------------------
def _state(x: float) -> dict:
    return {"objects": {"cube": {"pos": torch.tensor([x, 0.0, 0.0]), "rot": torch.tensor([1.0, 0, 0, 0])}}}


def test_hash_state_is_content_addressed_and_device_free():
    a = hash_state(_state(0.1))
    assert a == hash_state({"objects": {"cube": {"rot": [1.0, 0, 0, 0], "pos": [0.1, 0.0, 0.0]}}})
    assert a != hash_state(_state(0.2))
    assert hash_state(_state(0.1)) == hash_state({
        "objects": {"cube": {"pos": torch.tensor([0.1, 0, 0], dtype=torch.float64), "rot": [1, 0, 0, 0]}}
    })


def test_plan_instances_cycles_pool_and_seeds_per_episode():
    pool = [TaskInstance(index=i, seed=0, state=_state(i), state_hash=hash_state(_state(i))) for i in range(2)]
    plan = plan_instances(pool, EvalProtocol(episodes=5, seed=10))
    assert [p.index for p in plan] == [0, 1, 0, 1, 0]
    assert [p.seed for p in plan] == [10, 11, 12, 13, 14]
    assert plan[0].instance_id.startswith("0:10:") and plan[2].instance_id.startswith("0:12:")
    explicit = plan_instances(pool, EvalProtocol(instances=(1, 1)))
    assert [p.index for p in explicit] == [1, 1]
    with pytest.raises(ValueError, match="outside the pool"):
        plan_instances(pool, EvalProtocol(instances=(5,)))
    with pytest.raises(ValueError, match="no instance pool"):
        plan_instances([], EvalProtocol(instances=(0,)))
    seed_only = plan_instances([], EvalProtocol(episodes=2, seed=3))
    assert [(p.index, p.seed, p.state) for p in seed_only] == [(0, 3, None), (1, 4, None)]
    assert seed_only[0].instance_id == "0:3:default"


# ----------------------------------------------------------------------------------------------
# result
# ----------------------------------------------------------------------------------------------
def _episode(i: int, success: bool, steps: int = 10) -> EpisodeResult:
    return EpisodeResult(
        index=i,
        env_id=0,
        instance_index=i,
        instance_id=f"{i}:0:abc",
        seed=i,
        success=success,
        steps=steps,
        terminated=success,
        time_out=not success,
        return_sum=1.5,
    )


def test_wilson_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0) and 0.2 < high < 0.4
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0) and 0.6 < low < 0.8
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high


def test_result_roundtrip_summary_and_validation(tmp_path):
    r = BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=2,
        protocol=EvalProtocol(episodes=3).to_dict(),
        policy={"name": "hold", "kind": "HoldPolicy"},
        episodes=[_episode(0, True), _episode(1, False), _episode(2, True, steps=20)],
        provenance={"simulator": "mujoco"},
    )
    s = r.summary
    assert s["episodes"] == 3 and s["successes"] == 2 and s["success_rate"] == pytest.approx(2 / 3)
    assert s["mean_steps"] == pytest.approx(40 / 3) and s["time_outs"] == 1
    path = r.save(tmp_path / "sub" / "r.json")
    data = json.loads(path.read_text())
    assert data["schema"] == SCHEMA and data["schema_version"] == SCHEMA_VERSION and "summary" in data
    back = BenchmarkResult.load(path)
    assert back.episodes == r.episodes and back.policy == r.policy and back.summary == s

    bad = dict(data)
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        validate_result(bad)
    bad = dict(data)
    del bad["protocol"]
    with pytest.raises(ValueError, match="missing"):
        validate_result(bad)
    bad = json.loads(json.dumps(data))
    bad["episodes"][0]["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        validate_result(bad)


def test_summarize_pools_runs_with_the_same_key():
    proto = EvalProtocol(episodes=2).to_dict()
    mk = lambda eps, policy="hold", sim="mujoco": BenchmarkResult(  # noqa: E731
        task="t", simulator=sim, robots=["franka"], num_envs=1, protocol=proto, policy={"name": policy}, episodes=eps
    )
    table = summarize([
        mk([_episode(0, True), _episode(1, False)]),
        mk([_episode(0, True), _episode(1, True)]),
        mk([_episode(0, False)], policy="random"),
        mk([_episode(0, True)], sim="newton"),
    ])
    rows = [line for line in table.splitlines() if line.startswith("| t ")]
    assert len(rows) == 3
    pooled = next(r for r in rows if "| hold | 2 | 4 |" in r)
    assert "0.750" in pooled


# ----------------------------------------------------------------------------------------------
# evaluator, on a scripted env
# ----------------------------------------------------------------------------------------------
class _Handler:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.scenario = SimpleNamespace(simulator="fake", robots=[], objects=[], sim_params=None, decimation=1)

    def get_action_joint_names(self):
        return {"bot": ["j0", "j1"]}

    def get_states(self, env_ids=None, mode="tensor"):
        return "states"


class _ScriptedEnv:
    """Success at ``success_at`` steps into every episode, episodes end after ``length`` steps.

    ``schedule[instance_index] = (success_at | None, length, terminated_at_end)``.
    """

    auto_reset = False
    success_is_termination = False
    max_episode_steps = 1000
    traj_filepath = None
    task_desc = "do the thing"

    def __init__(self, num_envs, schedule, pool_size=0):
        self.num_envs = num_envs
        self.handler = _Handler(num_envs)
        self.device = torch.device("cpu")
        robot = SimpleNamespace(
            name="bot", joint_limits={"j0": (-1, 1), "j1": (0, 2)}, default_joint_positions={"j0": 0.5, "j1": 1.0}
        )
        self.scenario = SimpleNamespace(simulator="fake", robots=[robot])
        self.schedule = schedule
        self._initial_states = [_state(i) for i in range(pool_size)] if pool_size else None
        self.slot_instance = [None] * num_envs
        self.steps = torch.zeros(num_envs, dtype=torch.int32)
        self.resets: list[tuple[list[int], list | None, int | None]] = []
        self.actions = []

    actions_are_joint_targets = True

    def reset(self, states=None, env_ids=None, seed=None):
        env_ids = list(range(self.num_envs)) if env_ids is None else env_ids
        self.resets.append((list(env_ids), states, seed))
        for k, i in enumerate(env_ids):
            self.steps[i] = 0
            if states is not None:
                # the backends' rule: a full batch is indexed by env id, a partial one by position
                row = states[i] if len(states) == self.num_envs else states[k]
                self.slot_instance[i] = int(row["objects"]["cube"]["pos"][0].item())
            else:
                self.slot_instance[i] = 0
        return ("obs", len(env_ids)), {}  # BaseTaskEnv-style: only the reset rows

    def _observation(self, states):
        return ("obs", self.num_envs)

    def _entry(self, i):
        return self.schedule[self.slot_instance[i] % len(self.schedule)]

    def step(self, actions):
        self.actions.append(actions)
        self.steps += 1
        n = self.num_envs
        success = torch.zeros(n, dtype=torch.bool)
        terminated = torch.zeros(n, dtype=torch.bool)
        time_out = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            success_at, length, term_end = self._entry(i)
            s = int(self.steps[i])
            if success_at is not None and s >= success_at:
                success[i] = True
            if s >= length:
                (terminated if term_end else time_out)[i] = True
        done = terminated | time_out
        info = {"success": success.clone()}
        if self.auto_reset:
            for i in done.nonzero().flatten().tolist():
                self.reset(env_ids=[i])
        return ("obs", n), torch.ones(n), terminated, time_out, info


def test_evaluate_scores_each_planned_instance_once_in_order():
    # instance 0: success from step 3, ends at 5 (terminated); instance 1: never, times out at 4
    env = _ScriptedEnv(num_envs=2, schedule=[(3, 5, True), (None, 4, False)], pool_size=2)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=4, seed=100), task_name="scripted")
    assert result.task == "scripted" and result.simulator == "fake" and result.robots == ["bot"]
    assert [e.index for e in result.episodes] == [0, 1, 2, 3]
    assert [e.instance_index for e in result.episodes] == [0, 1, 0, 1]
    assert [e.success for e in result.episodes] == [True, False, True, False]
    assert [e.steps for e in result.episodes] == [5, 4, 5, 4]
    assert [e.terminated for e in result.episodes] == [True, False, True, False]
    assert [e.time_out for e in result.episodes] == [False, True, False, True]
    assert [e.first_success_step for e in result.episodes] == [3, None, 3, None]
    # episodes 0 and 1 started in one batched reset and share its seed (the handler RNG is global);
    # instance 1 is shorter, so episode 2 lands on env 1 and episode 3 on env 0, each seeded on its own
    assert [e.seed for e in result.episodes] == [100, 100, 102, 103]
    assert result.summary["success_rate"] == 0.5 and result.notes["instance_pool"] == 2
    assert result.notes["instruction"] == "do the thing"
    assert [(ids, seed) for ids, _, seed in env.resets] == [([0, 1], 100), ([1], 102), ([0], 103)]
    assert [e.env_id for e in result.episodes] == [0, 1, 1, 0]
    # explicit states always travel as a full batch (row k = env k), whatever subset is written
    assert all(states is not None and len(states) == 2 for _, states, _ in env.resets)
    # every hold action covered both env slots with the default joint positions
    assert all(a.shape == (2, 2) and torch.equal(a[0], torch.tensor([0.5, 1.0])) for a in env.actions)


def test_evaluate_final_and_held_rules_change_the_verdict():
    env = _ScriptedEnv(num_envs=1, schedule=[(2, 3, False)], pool_size=1)  # success at steps 2,3; times out at 3
    assert evaluate(env, HoldPolicy(), EvalProtocol(episodes=1, success_rule="final")).episodes[0].success
    env = _ScriptedEnv(num_envs=1, schedule=[(2, 3, False)], pool_size=1)
    assert (
        not evaluate(env, HoldPolicy(), EvalProtocol(episodes=1, success_rule="held", hold_steps=3)).episodes[0].success
    )
    env = _ScriptedEnv(num_envs=1, schedule=[(2, 3, False)], pool_size=1)
    assert evaluate(env, HoldPolicy(), EvalProtocol(episodes=1, success_rule="held", hold_steps=2)).episodes[0].success


def test_evaluate_cap_ends_episode_as_time_out_and_resets():
    env = _ScriptedEnv(num_envs=1, schedule=[(None, 50, True)], pool_size=1)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2, max_episode_steps=7))
    assert [(e.steps, e.time_out, e.terminated) for e in result.episodes] == [(7, True, False), (7, True, False)]
    assert [seed for _, _, seed in env.resets] == [0, 1]  # initial + one between the episodes, each seeded


def test_evaluate_auto_reset_env_episodes_are_seeded_by_the_evaluator():
    env = _ScriptedEnv(num_envs=2, schedule=[(1, 2, True)], pool_size=0)
    env.auto_reset = True
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=4, seed=10))
    assert all(e.success for e in result.episodes) and len(result.episodes) == 4
    # both slots finish on the same step, so each pair of episodes starts in one seeded reset
    assert [e.seed for e in result.episodes] == [10, 10, 12, 12]
    # the env's own auto-resets carry no seed; the evaluator re-seeds every batch of started episodes
    seeded = [(ids, seed) for ids, _, seed in env.resets if seed is not None]
    assert seeded == [([0, 1], 10), ([0, 1], 12)]


def test_evaluate_auto_reset_env_gets_explicit_instance_state_after_its_own_reset():
    env = _ScriptedEnv(num_envs=1, schedule=[(1, 2, True), (None, 3, False)], pool_size=2)
    env.auto_reset = True
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2))
    assert [e.instance_index for e in result.episodes] == [0, 1]
    assert [e.success for e in result.episodes] == [True, False]
    # the second instance's state was written after the env's auto-reset, seeded with its episode seed
    explicit = [r for r in env.resets if r[1] is not None]
    assert len(explicit) == 2 and explicit[1][0] == [0] and explicit[1][2] == 1


def test_evaluate_env_whose_reset_takes_no_states():
    """The manager-based RL env resets with ``(env_ids, seed)`` only; default-reset instances must still work."""

    class _NoStates(_ScriptedEnv):
        auto_reset = True

        def reset(self, env_ids=None, seed=None):
            return super().reset(None, env_ids, seed)

    env = _NoStates(num_envs=2, schedule=[(1, 2, True)], pool_size=0)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=3))
    assert len(result.episodes) == 3 and all(e.success for e in result.episodes)
    # an explicit instance state cannot be honoured by such an env: say so instead of guessing
    env = _NoStates(num_envs=1, schedule=[(1, 2, True)], pool_size=1)
    with pytest.raises(TypeError, match="cannot start from an explicit task instance"):
        evaluate(env, HoldPolicy(), EvalProtocol(episodes=1))


def test_evaluate_idle_slots_get_default_resets_and_are_not_scored():
    env = _ScriptedEnv(num_envs=4, schedule=[(1, 2, True), (None, 3, False)], pool_size=2)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2))
    assert [e.instance_index for e in result.episodes] == [0, 1]
    first, second = env.resets[0], env.resets[1]
    assert first[0] == [0, 1] and len(first[1]) == 4  # the two planned slots, from a full 4-row batch
    assert second[0] == [2, 3] and second[1] is None  # idle slots: default reset, same call sequence
    # a slot that finishes with nothing left to run is put back to the default state, not left mid-episode
    assert all(states is None for _, states, _ in env.resets[2:])


def test_context_uses_the_env_action_space_only_when_the_env_maps_actions_itself():
    import gymnasium as gym
    import numpy as np

    from metasim.bench import build_context
    from metasim.task.base import BaseTaskEnv
    from metasim.task.rl_task import RLTaskEnv

    assert BaseTaskEnv.actions_are_joint_targets is True and RLTaskEnv.actions_are_joint_targets is True

    env = _ScriptedEnv(num_envs=1, schedule=[(None, 2, False)])
    env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    ctx = build_context(env, EvalProtocol())  # joint targets: the declared box is ignored
    assert ctx.action_dim == 2 and ctx.default_action.tolist() == [0.5, 1.0]
    env.actions_are_joint_targets = False  # an action manager owns the mapping: the box is the contract
    ctx = build_context(env, EvalProtocol())
    assert ctx.action_dim == 1 and ctx.default_action.tolist() == [0.0]
    assert ctx.action_low.tolist() == [-1.0] and ctx.action_high.tolist() == [1.0]
    assert ctx.default_joint_positions == {"bot": {"j0": 0.5, "j1": 1.0}}
    env.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))  # BaseTaskEnv's default: unusable
    assert build_context(env, EvalProtocol()).action_dim == 2


def test_policy_always_sees_a_full_batch_observation():
    """Two reset calls in one start (explicit + default slots) must not leak a partial observation."""
    seen = []

    class _Spy(HoldPolicy):
        def act(self, obs, ctx):
            seen.append(obs)
            return super().act(obs, ctx)

    # both envs time out together; the one remaining instance restarts env 0 explicitly, env 1 idles
    env = _ScriptedEnv(num_envs=2, schedule=[(None, 2, False)], pool_size=1)
    evaluate(env, _Spy(), EvalProtocol(episodes=3))
    assert seen and all(obs == ("obs", 2) for obs in seen)


def test_summary_and_table_say_undefined_without_a_success_signal():
    eps = [_episode(0, False), _episode(1, False)]
    r = BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=[],
        num_envs=1,
        protocol=EvalProtocol().to_dict(),
        policy={"name": "random"},
        episodes=eps,
        notes={"success_defined": False},
    )
    assert r.summary["success_defined"] is False and r.summary["success_rate"] is None
    row = next(line for line in summarize([r]).splitlines() if line.startswith("| t "))
    assert "| undefined |" in row and "0.000" not in row
    r.notes["success_defined"] = True
    assert r.summary["success_rate"] == 0.0


def test_protocol_episodes_default_and_explicit_instances():
    assert EvalProtocol().episodes == 10
    assert EvalProtocol(instances=(3, 1)).episodes == 2
    assert EvalProtocol(episodes=2, instances=(3, 1)).episodes == 2
    with pytest.raises(ValueError, match="disagrees"):
        EvalProtocol(episodes=10, instances=(0, 1))
    assert EvalProtocol.from_dict(EvalProtocol(instances=(0,)).to_dict()).episodes == 1


def test_validate_rejects_unknown_top_level_keys():
    r = BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=[],
        num_envs=1,
        protocol=EvalProtocol().to_dict(),
        policy={"name": "p"},
        episodes=[_episode(0, True)],
    )
    data = r.to_dict()
    data["comment"] = "hand-written"
    with pytest.raises(ValueError, match="unknown top-level keys"):
        validate_result(data)


def test_step_to_host_packs_tensors_and_accepts_scalars():
    from metasim.bench.evaluator import _step_to_host

    t = torch.tensor([True, False])
    out = _step_to_host(t, ~t, t, torch.tensor([1.5, 2.5]), 2)
    assert out == ([True, False], [False, True], [True, False], [1.5, 2.5])
    assert _step_to_host(True, False, False, 1.0, 2) == ([True, True], [False, False], [False, False], [1.0, 1.0])


def test_instance_pool_reads_an_rl_tensor_state_batch(monkeypatch):
    from metasim.bench import instance_pool
    from metasim.utils import state as state_mod

    env = _ScriptedEnv(num_envs=2, schedule=[(1, 2, True)], pool_size=0)
    env._initial_states = SimpleNamespace(robots={"bot": object()}, objects={})  # stands in for a TensorState
    monkeypatch.setattr(state_mod, "state_tensor_to_nested", lambda handler, ts: [_state(0), _state(1), _state(0)])
    pool = instance_pool(env)
    assert [p.index for p in pool] == [0, 1]
    env = _ScriptedEnv(num_envs=2, schedule=[(1, 2, True)], pool_size=0)
    env._initial_states = SimpleNamespace(robots={}, objects={})  # a scene-only task: nothing to draw from
    assert instance_pool(env) == []


def test_evaluate_without_success_info_falls_back_to_the_termination_reading():
    class _NoInfo(_ScriptedEnv):
        success_is_termination = True

        def step(self, actions):
            obs, r, t, to, _ = super().step(actions)
            return obs, r, t, to, {}

    env = _NoInfo(num_envs=1, schedule=[(None, 2, True), (None, 2, False)], pool_size=2)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2))
    assert [e.success for e in result.episodes] == [True, False]


def test_replay_policy_feeds_instance_actions_then_holds(monkeypatch):
    pool_actions = [[{"bot": {"dof_pos_target": {"j0": 0.1 * t, "j1": 0.0}}} for t in range(3)]]
    env = _ScriptedEnv(num_envs=2, schedule=[(None, 5, False)], pool_size=1)
    inst = TaskInstance(index=0, seed=0, state=_state(0), actions=pool_actions[0], state_hash="h")
    env._initial_states = None
    from metasim.bench import evaluator

    monkeypatch.setattr(evaluator, "instance_pool", lambda e: [inst])
    result = evaluate(env, ReplayPolicy(), EvalProtocol(episodes=1))
    assert len(result.episodes) == 1 and result.episodes[0].steps == 5
    first = env.actions[0]
    assert first[0]["bot"]["dof_pos_target"]["j0"] == 0.0 and first[1]["bot"]["dof_pos_target"] == {
        "j0": 0.5,
        "j1": 1.0,
    }
    assert env.actions[2][0]["bot"]["dof_pos_target"]["j0"] == pytest.approx(0.2)
    assert env.actions[4][0]["bot"]["dof_pos_target"]["j0"] == pytest.approx(0.2)  # held after the demo ends


def test_random_policy_is_reproducible_and_within_limits():
    ctx = PolicyContext(
        task="t",
        simulator="fake",
        robots=["bot"],
        num_envs=3,
        device=torch.device("cpu"),
        seed=5,
        max_episode_steps=10,
        action_joint_names={"bot": ["j0", "j1"]},
        action_low=torch.tensor([-1.0, 0.0]),
        action_high=torch.tensor([1.0, 2.0]),
        default_action=torch.tensor([0.0, 1.0]),
    )
    a, b = RandomPolicy(), RandomPolicy()
    a.reset([0], [], ctx)
    b.reset([0], [], ctx)
    x, y = a.act(None, ctx), b.act(None, ctx)
    assert torch.equal(x, y) and x.shape == (3, 2)
    assert (x[:, 0] >= -1).all() and (x[:, 0] <= 1).all() and (x[:, 1] >= 0).all() and (x[:, 1] <= 2).all()
    with pytest.raises(RuntimeError, match="before reset"):
        RandomPolicy().act(None, ctx)


def test_load_policy_builtin_and_import_path():
    assert isinstance(load_policy("hold"), HoldPolicy)
    assert isinstance(load_policy("random", seed=3), RandomPolicy)
    assert isinstance(load_policy("metasim.bench.policy:HoldPolicy"), HoldPolicy)
    with pytest.raises(ValueError, match="unknown policy"):
        load_policy("nope")
    with pytest.raises(ValueError, match="no attribute"):
        load_policy("metasim.bench.policy:Missing")
    with pytest.raises(TypeError, match="does not satisfy"):
        load_policy("metasim.bench.policy:checkpoint_sha256", path=None)


def test_task_env_step_reports_success_by_class_flag():
    from metasim.task.base import BaseTaskEnv
    from metasim.task.rl_task import RLTaskEnv

    assert BaseTaskEnv.success_is_termination is True and BaseTaskEnv.auto_reset is False
    assert RLTaskEnv.success_is_termination is False and RLTaskEnv.auto_reset is True
    env = BaseTaskEnv.__new__(BaseTaskEnv)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.handler = SimpleNamespace(num_envs=2)
    terminated = torch.tensor([True, False])
    assert torch.equal(env._step_success("states", terminated), terminated)

    class _Separate(BaseTaskEnv):
        success_is_termination = False

        def _success(self, env_states):
            return [False, True]

    env = _Separate.__new__(_Separate)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.handler = SimpleNamespace(num_envs=2)
    assert env._step_success("states", terminated).tolist() == [False, True]
    assert env.defines_success is True
    assert BaseTaskEnv.__new__(BaseTaskEnv).defines_success is True  # terminated is read as success
    assert RLTaskEnv.__new__(RLTaskEnv).defines_success is False  # neither flag nor override


def test_evaluate_records_whether_the_task_defines_success():
    env = _ScriptedEnv(num_envs=1, schedule=[(1, 2, True)], pool_size=1)
    env.defines_success = False
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=1))
    assert result.notes["success_defined"] is False
    env = _ScriptedEnv(num_envs=1, schedule=[(1, 2, True)], pool_size=1)
    assert evaluate(env, HoldPolicy(), EvalProtocol(episodes=1)).notes["success_defined"] is True


def test_cli_constants_mirror_the_bench_package():
    from metasim import __main__ as cli
    from metasim.bench import BUILTIN_POLICIES, SUCCESS_RULES

    assert cli._SUCCESS_RULES == SUCCESS_RULES
    assert tuple(cli._BUILTIN_POLICIES) == tuple(BUILTIN_POLICIES)


def test_evaluate_broadcasts_a_scalar_reward_and_terminated():
    class _ScalarAll(_ScriptedEnv):
        def step(self, actions):
            obs, r, t, to, info = super().step(actions)
            return obs, 1.0, bool(t[0]), bool(to[0]), {"success": bool(info["success"][0])}

    env = _ScalarAll(num_envs=1, schedule=[(2, 3, True)], pool_size=1)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=1))
    assert result.episodes[0].success and result.episodes[0].return_sum == 3.0 and result.episodes[0].terminated


def test_evaluate_broadcasts_a_scalar_success_flag():
    """Native wrappers (robosuite, LIBERO) report ``{"success": bool}`` for their single env."""

    class _Scalar(_ScriptedEnv):
        def step(self, actions):
            obs, r, t, to, info = super().step(actions)
            info["success"] = bool(info["success"][0])
            return obs, r, t, to, info

    env = _Scalar(num_envs=1, schedule=[(2, 3, True)], pool_size=1)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=1))
    assert result.episodes[0].success and result.episodes[0].first_success_step == 2


def test_hold_dict_action_uses_joint_defaults_even_with_a_reduced_action_box():
    from metasim.bench.policy import _hold_dict_action

    ctx = PolicyContext(
        task="t",
        simulator="fake",
        robots=["bot"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=1,
        action_joint_names={"bot": ["j0", "j1"]},
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        default_action=torch.tensor([0.0]),
        default_joint_positions={"bot": {"j0": 0.5, "j1": 1.0}},
    )
    assert _hold_dict_action(ctx) == {"bot": {"dof_pos_target": {"j0": 0.5, "j1": 1.0}}}


def test_instance_pool_dedupes_tiled_initial_states():
    from metasim.bench import instance_pool

    env = _ScriptedEnv(num_envs=4, schedule=[(1, 2, True)], pool_size=0)
    env._initial_states = [_state(0), _state(1), _state(0), _state(1)]  # a task base tiled 2 demos to 4 envs
    pool = instance_pool(env)
    assert [p.index for p in pool] == [0, 1] and len({p.state_hash for p in pool}) == 2
    assert instance_pool(env) == pool  # cached on the env


def test_random_policy_refuses_unbounded_joints():
    ctx = PolicyContext(
        task="t",
        simulator="fake",
        robots=["bot"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=1,
        action_joint_names={"bot": ["j0"]},
        action_low=torch.tensor([-float("inf")]),
        action_high=torch.tensor([float("inf")]),
        default_action=torch.tensor([0.0]),
    )
    policy = RandomPolicy()
    policy.reset([0], [], ctx)
    with pytest.raises(ValueError, match="finite action bounds"):
        policy.act(None, ctx)


def test_reset_kwargs_come_from_the_signature_not_the_error_text():
    from metasim.bench.evaluator import _reset

    class _Buggy(_ScriptedEnv):
        def reset(self, states=None, env_ids=None, seed=None):
            raise TypeError("unsupported operand for states and seed")  # a real bug inside reset

    env = _Buggy(num_envs=1, schedule=[(1, 2, True)], pool_size=1)
    with pytest.raises(TypeError, match="unsupported operand"):
        _reset(env, [0], [_state(0)], 0)


def test_step_success_does_not_alias_terminated():
    from metasim.task.base import BaseTaskEnv

    env = BaseTaskEnv.__new__(BaseTaskEnv)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.handler = SimpleNamespace(num_envs=2)
    terminated = torch.tensor([False, False])
    success = env._step_success("states", terminated)
    terminated |= torch.tensor([True, True])  # a learner folding time-outs into done, in place
    assert success.tolist() == [False, False]


def test_summarize_keeps_hand_picked_instances_apart():
    eps = [_episode(0, True)]
    pooled = BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=1,
        protocol=EvalProtocol(episodes=1).to_dict(),
        policy={"name": "p"},
        episodes=eps,
    )
    picked = BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=1,
        protocol=EvalProtocol(instances=(0,)).to_dict(),
        policy={"name": "p"},
        episodes=eps,
    )
    rows = [line for line in summarize([pooled, picked]).splitlines() if line.startswith("| t ")]
    assert len(rows) == 2 and any("instances [0]" in r for r in rows)


def test_health_task_classifies_outcomes(monkeypatch, tmp_path):
    """One row per task: ok / low / undefined by replay success, unsupported, no-demos, and error for a crash."""
    from types import SimpleNamespace

    import metasim.__main__ as cli
    from metasim.task import registry

    args = SimpleNamespace(
        sim="mujoco",
        task=None,
        robot=None,
        num_envs=1,
        episodes=2,
        max_steps=None,
        seed=0,
        threshold=0.8,
        device="cpu",
        out_dir=str(tmp_path),
    )

    class _Env(_ScriptedEnv):
        def __init__(self, scenario, device):
            super().__init__(num_envs=1, schedule=[(None, 2, False)], pool_size=0)
            self.scenario = scenario

        def close(self):
            return None

    def task_class(name):
        cls = type(name, (_Env,), {"scenario": SimpleNamespace(replace=lambda **kw: SimpleNamespace(**kw))})
        if name == "fresh":
            cls.traj_filepath = None
        else:
            cls.traj_filepath = "demo.pkl"
        if name == "newton_only":
            cls.supported_simulators = ("newton",)
        if name == "broken":
            cls.__init__ = lambda self, scenario, device: (_ for _ in ()).throw(RuntimeError("asset missing"))
        return cls

    monkeypatch.setattr(registry, "get_task_class", task_class)

    results = {
        "healthy": [_episode(0, True), _episode(1, True)],
        "sick": [_episode(0, False), _episode(1, True)],
        "mute": [_episode(0, False), _episode(1, False)],
    }

    def fake_evaluate(env, policy, protocol, policy_info=None, on_episode=None, task_name=None):
        return BenchmarkResult(
            task=task_name,
            simulator="mujoco",
            robots=["bot"],
            num_envs=1,
            protocol=protocol.to_dict(),
            policy={"name": policy.name},
            episodes=results[task_name],
            notes={"success_defined": task_name != "mute"},
        )

    import metasim.bench as bench

    monkeypatch.setattr(bench, "evaluate", fake_evaluate)

    rows = {
        name: cli._health_task(name, args) for name in ("healthy", "sick", "mute", "newton_only", "fresh", "broken")
    }
    assert rows["healthy"]["status"] == "ok" and rows["healthy"]["success_rate"] == 1.0
    assert rows["sick"]["status"] == "low" and rows["sick"]["success_rate"] == 0.5
    assert rows["mute"]["status"] == "undefined" and rows["mute"]["success_rate"] is None
    assert rows["newton_only"]["status"] == "unsupported" and "newton" in rows["newton_only"]["detail"]
    assert rows["fresh"]["status"] == "no-demos"
    assert rows["broken"]["status"] == "error" and "asset missing" in rows["broken"]["detail"]
    assert (tmp_path / "healthy__bot__mujoco__replay.json").exists(), "a health run leaves ordinary result files"

    table = cli._health_table(list(rows.values()), 0.8)
    lines = table.splitlines()
    assert lines[2].startswith("| healthy | ok | 1.000 |") and lines[3].startswith("| sick | low | 0.500 |")
    assert "1 ok, 1 low, 1 undefined, 1 unsupported, 1 no-demos, 1 error" in lines[-1]


def test_bench_health_gates_on_low_and_error(monkeypatch, tmp_path):
    import metasim.__main__ as cli
    from metasim.task import registry

    monkeypatch.setattr(registry, "list_tasks", lambda: ["a", "b", "c"])
    canned = {
        "a": {"task": "a", "status": "ok", "success_rate": 1.0, "episodes": 2, "result": None, "detail": ""},
        "b": {"task": "b", "status": "no-demos", "success_rate": None, "episodes": 0, "result": None, "detail": ""},
        "c": {"task": "c", "status": "low", "success_rate": 0.0, "episodes": 2, "result": None, "detail": ""},
    }
    monkeypatch.setattr(cli, "_health_task", lambda name, args: canned[name])

    out = tmp_path / "health"
    code = cli.main(["bench", "health", "--sim", "mujoco", "--out-dir", str(out), "--episodes", "2"])
    assert code == 1, "a low task fails the gate"
    data = json.loads((out / "health.json").read_text())
    assert [r["task"] for r in data["rows"]] == ["a", "c"], "the default sweep skips tasks without demos"
    assert (out / "health.md").read_text().startswith("| task | status |")

    canned["c"]["status"] = "ok"
    assert cli.main(["bench", "health", "--sim", "mujoco", "--task", "b", "--task", "c", "--out-dir", str(out)]) == 0
    data = json.loads((out / "health.json").read_text())
    assert [r["task"] for r in data["rows"]] == ["b", "c"], "an explicit --task is always reported"
