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

    def reset(self, states=None, env_ids=None, seed=None):
        env_ids = list(range(self.num_envs)) if env_ids is None else env_ids
        self.resets.append((list(env_ids), states, seed))
        for k, i in enumerate(env_ids):
            self.steps[i] = 0
            if states is not None:
                self.slot_instance[i] = int(states[k]["objects"]["cube"]["pos"][0].item())
            else:
                self.slot_instance[i] = 0
        return "obs", {}

    def _observation(self, states):
        return "obs"

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
        return "obs", torch.ones(n), terminated, time_out, info


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
    assert [e.seed for e in result.episodes] == [100, 101, 102, 103]
    assert result.summary["success_rate"] == 0.5 and result.notes["instance_pool"] == 2
    assert result.notes["instruction"] == "do the thing"
    # the first reset wrote both planned instances explicitly, with their states
    ids, states, seed = env.resets[0]
    assert ids == [0, 1] and len(states) == 2 and seed == 100
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
    assert len(env.resets) == 2  # initial + one between the episodes


def test_evaluate_auto_reset_env_is_not_reset_twice_without_explicit_state():
    env = _ScriptedEnv(num_envs=2, schedule=[(1, 2, True)], pool_size=0)
    env.auto_reset = True
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=4))
    assert all(e.success for e in result.episodes) and len(result.episodes) == 4
    # initial reset by the evaluator (2 slots, default states), then only the env's own auto-resets
    evaluator_resets = [r for r in env.resets if r[2] is not None]
    assert len(evaluator_resets) == 1 and evaluator_resets[0][1] is None


def test_evaluate_auto_reset_env_gets_explicit_instance_state_after_its_own_reset():
    env = _ScriptedEnv(num_envs=1, schedule=[(1, 2, True), (None, 3, False)], pool_size=2)
    env.auto_reset = True
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2))
    assert [e.instance_index for e in result.episodes] == [0, 1]
    assert [e.success for e in result.episodes] == [True, False]
    # the second instance's state was written after the env's auto-reset
    explicit = [r for r in env.resets if r[1] is not None]
    assert len(explicit) == 2 and explicit[1][0] == [0]


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
