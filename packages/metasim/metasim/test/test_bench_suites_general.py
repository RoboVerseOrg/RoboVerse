"""``metasim.bench`` suites, prompt banks, demo budgets and served policies, without a simulator.

Reuses the scripted env of ``test_bench_general`` so the evaluator's counting is the one already
pinned there; what is new here is which instruction and cap each episode gets, how suite results
aggregate, and that a policy behind :class:`PolicyServer` scores exactly like the same policy in-process.
"""

from __future__ import annotations

import threading
from collections import Counter

import numpy as np
import pytest
import torch

from metasim.bench import (
    BenchmarkSuite,
    EvalProtocol,
    HoldPolicy,
    PolicyServer,
    PromptBank,
    RemotePolicy,
    SuiteTask,
    TaskInstance,
    demo_budget,
    evaluate,
    plan_instances,
    suite_report,
    suite_table,
    task_prompt_bank,
)
from metasim.bench import suite as suite_mod
from metasim.bench.remote import actions_to_array, flatten_observation, unflatten_observation
from metasim.test.test_bench_general import _episode, _ScriptedEnv, _state

pytestmark = pytest.mark.general

BANK = PromptBank(
    canonical="stack the red cube on the blue cube",
    train=tuple(f"train paraphrase {i}" for i in range(8)),
    heldout=("put red onto blue", "place the red block on top of the blue one"),
)


# ----------------------------------------------------------------------------------------------
# prompt banks
# ----------------------------------------------------------------------------------------------
def test_prompt_draws_are_deterministic_and_respect_the_split():
    draws = [BANK.draw(task="t", episode=i, seed=i, split="train") for i in range(400)]
    assert draws == [BANK.draw(task="t", episode=i, seed=i, split="train") for i in range(400)]
    assert all(d.prompt not in BANK.heldout for d in draws)
    variants = Counter(d.variant for d in draws)
    assert 0.12 < variants["canonical"] / 400 < 0.28  # canonical_probability 0.2
    assert len([v for v in variants if v.startswith("train_")]) == 8
    held = {BANK.draw(task="t", episode=i, seed=i, split="heldout").prompt for i in range(50)}
    assert held == set(BANK.heldout)
    assert BANK.draw(task="t", episode=3, seed=3).prompt == BANK.canonical  # default split
    # a different task draws a different sequence
    other = [BANK.draw(task="u", episode=i, seed=i, split="train").variant for i in range(50)]
    assert other != [d.variant for d in draws[:50]]


def test_prompt_bank_refuses_leaks_and_empty_heldout():
    with pytest.raises(ValueError, match="training set"):
        PromptBank(canonical="a", train=("b",), heldout=("b",))
    with pytest.raises(ValueError, match="training set"):
        PromptBank(canonical="a", heldout=("a",))
    with pytest.raises(ValueError, match="non-empty"):
        PromptBank(canonical=" ")
    with pytest.raises(ValueError, match="no held-out"):
        PromptBank(canonical="a").draw(task="t", episode=0, seed=0, split="heldout")
    assert PromptBank(canonical="a").draw(task="t", episode=0, seed=0, split="train").prompt == "a"


def test_task_prompt_bank_falls_back_to_task_desc():
    env = _ScriptedEnv(num_envs=1, schedule=[(None, 2, False)])
    assert task_prompt_bank(env) == PromptBank(canonical="do the thing")
    env.prompt_bank = BANK
    assert task_prompt_bank(env) is BANK
    env.prompt_bank = {"canonical": "x"}
    with pytest.raises(TypeError):
        task_prompt_bank(env)


def test_plan_instances_draws_one_prompt_per_episode():
    pool = [TaskInstance(index=i, seed=0, state=_state(i)) for i in range(2)]
    plan = plan_instances(pool, EvalProtocol(episodes=6, seed=10, prompt_split="heldout"), prompts=BANK, task="t")
    assert [p.prompt for p in plan] == [
        BANK.draw(task="t", episode=i, seed=10 + i, split="heldout").prompt for i in range(6)
    ]
    assert all(p.prompt_variant.startswith("heldout_") for p in plan)
    with pytest.raises(ValueError, match="no prompt bank"):
        plan_instances(pool, EvalProtocol(episodes=1, prompt_split="heldout"), prompts=None, task="t")


def test_evaluate_records_each_episode_prompt():
    env = _ScriptedEnv(num_envs=2, schedule=[(1, 2, True)], pool_size=2)
    env.prompt_bank = BANK
    seen = []

    class Spy(HoldPolicy):
        def reset(self, env_ids, instances, ctx):
            seen.extend(i.prompt for i in instances)

    result = evaluate(env, Spy(), EvalProtocol(episodes=4, prompt_split="train"), task_name="t")
    assert [e.prompt for e in result.episodes] == seen
    assert all(e.prompt_variant for e in result.episodes)
    assert result.notes["prompt_draw_revision"].startswith("roboverse.bench.prompts/")
    roundtrip = type(result).from_dict(result.to_dict())
    assert [e.prompt for e in roundtrip.episodes] == seen


# ----------------------------------------------------------------------------------------------
# demo budgets
# ----------------------------------------------------------------------------------------------
def test_demo_budget_is_factor_times_the_quantile_of_demo_lengths():
    pool = [TaskInstance(index=i, seed=0, actions=[0] * n) for i, n in enumerate([10, 20, 30, 40, 100])]
    assert demo_budget(pool, factor=3.0) == int(np.ceil(3 * np.quantile([10, 20, 30, 40, 100], 0.75)))
    assert demo_budget(pool, factor=1.0, quantile=1.0) == 100
    assert demo_budget([TaskInstance(index=0, seed=0)], factor=3.0) is None


def test_protocol_refuses_two_caps_and_bad_budget_values():
    with pytest.raises(ValueError, match="not both"):
        EvalProtocol(max_episode_steps=10, demo_budget_factor=3.0)
    with pytest.raises(ValueError, match="> 0"):
        EvalProtocol(demo_budget_factor=0.0)
    with pytest.raises(ValueError, match="quantile"):
        EvalProtocol(demo_budget_factor=2.0, demo_budget_quantile=0.0)
    with pytest.raises(ValueError, match="prompt_split"):
        EvalProtocol(prompt_split="test")
    p = EvalProtocol(demo_budget_factor=3.0, prompt_split="heldout")
    assert EvalProtocol.from_dict(p.to_dict()) == p
    # results written before the fields existed still load
    legacy = {
        k: v for k, v in p.to_dict().items() if k not in ("demo_budget_factor", "demo_budget_quantile", "prompt_split")
    }
    assert EvalProtocol.from_dict(legacy) == EvalProtocol()


def test_evaluate_applies_the_demo_budget(monkeypatch):
    env = _ScriptedEnv(num_envs=1, schedule=[(None, 500, False)])
    pool = [
        TaskInstance(index=i, seed=0, state=_state(i), actions=[{}] * n, state_hash=str(i))
        for i, n in enumerate([4, 4, 6])
    ]
    from metasim.bench import evaluator

    monkeypatch.setattr(evaluator, "instance_pool", lambda e: pool)
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2, demo_budget_factor=2.0), task_name="t")
    cap = int(np.ceil(2.0 * np.quantile([4, 4, 6], 0.75)))
    assert [(e.steps, e.time_out) for e in result.episodes] == [(cap, True), (cap, True)]
    assert result.notes["episode_cap"] == cap and result.notes["episode_cap_source"] == "demos"
    # a budget above the task's own time-out is clipped to it, and the notes say so
    env = _ScriptedEnv(num_envs=1, schedule=[(None, 500, False)])
    env.max_episode_steps = 5
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=1, demo_budget_factor=2.0), task_name="t")
    assert result.notes["episode_cap"] == 5 and result.notes["episode_cap_source"].startswith("task")
    # no demos: the task's own cap applies and the notes say so
    monkeypatch.setattr(evaluator, "instance_pool", lambda e: [])
    env = _ScriptedEnv(num_envs=1, schedule=[(None, 3, False)])
    result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=1, demo_budget_factor=2.0), task_name="t")
    assert result.episodes[0].steps == 3 and result.notes["episode_cap_source"] == "task"


# ----------------------------------------------------------------------------------------------
# suites
# ----------------------------------------------------------------------------------------------
def _result(task, successes, n, *, robots=("bot",), simulator="fake", notes=None):
    from metasim.bench import BenchmarkResult

    return BenchmarkResult(
        task=task,
        simulator=simulator,
        robots=list(robots),
        num_envs=1,
        protocol=EvalProtocol(episodes=n).to_dict(),
        policy={"name": "p"},
        episodes=[_episode(i, i < successes) for i in range(n)],
        notes=dict(notes or {}),
    )


SUITE = BenchmarkSuite(
    name="toy",
    splits={"core": ("a", "b"), "heldout": (SuiteTask(task="c", protocol={"episodes": 5}),)},
    protocol=EvalProtocol(episodes=10, demo_budget_factor=3.0),
    simulator="fake",
)


def test_suite_validates_and_applies_overrides():
    assert [(s, t.task) for s, t in SUITE.entries()] == [("core", "a"), ("core", "b"), ("heldout", "c")]
    assert SUITE.protocol_for(SUITE.splits["heldout"][0]).episodes == 5
    assert SUITE.protocol_for(SUITE.splits["core"][0]).name == "toy"
    with pytest.raises(ValueError, match="both"):
        BenchmarkSuite(name="x", splits={"s1": ("a",), "s2": ("a",)})
    with pytest.raises(ValueError, match="unknown protocol fields"):
        BenchmarkSuite(name="x", splits={"s": (SuiteTask(task="a", protocol={"episode": 3}),)})
    with pytest.raises(ValueError, match="no split"):
        SUITE.entries("val")


def test_suite_report_is_task_macro_and_flags_missing_tasks():
    # a: 9/10, b: 1/2 -> macro 0.7, pooled 10/12
    report = suite_report(SUITE, [_result("a", 9, 10), _result("b", 1, 2), _result("zzz", 1, 1)])
    core, held = report["splits"]["core"], report["splits"]["heldout"]
    assert core["complete"] and core["macro_success_rate"] == pytest.approx(0.7)
    assert core["pooled_success_rate"] == pytest.approx(10 / 12)
    assert not held["complete"] and held["macro_success_rate"] is None and held["missing"] == ["c"]
    assert "INCOMPLETE" in suite_table(report) and "task-macro 0.700" in suite_table(report)
    # runs of one task are pooled; results tagged for another suite/split do not count
    report = suite_report(
        SUITE,
        [
            _result("a", 1, 2),
            _result("a", 1, 2),
            _result("b", 2, 2, notes={"suite": {"name": "other", "split": "core"}}),
        ],
        split="core",
    )
    rows = {r["task"]: r for r in report["splits"]["core"]["tasks"]}
    assert rows["a"]["runs"] == 2 and rows["a"]["success_rate"] == 0.5 and rows["b"]["status"] == "missing"
    with pytest.raises(ValueError, match="several simulators"):
        suite_report(SUITE, [_result("a", 1, 1), _result("b", 1, 1, simulator="other")])


def test_suite_report_marks_undefined_success():
    report = suite_report(
        SUITE, [_result("a", 0, 3, notes={"success_defined": False}), _result("b", 1, 1)], split="core"
    )
    rows = {r["task"]: r for r in report["splits"]["core"]["tasks"]}
    assert rows["a"]["status"] == "undefined" and not report["splits"]["core"]["complete"]


def test_suite_registry_discovers_content_packages(monkeypatch, tmp_path):
    pkg = tmp_path / "toy_suites_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from metasim.bench import BenchmarkSuite, register_suite\n"
        "register_suite(BenchmarkSuite(name='discovered', splits={'core': ('x',)}))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("METASIM_SUITE_PACKAGES", "toy_suites_pkg:missing_suites_pkg")
    suite_mod._reset_registry_for_tests()
    try:
        assert "discovered" in suite_mod.list_suites()
        assert suite_mod.get_suite("discovered").splits["core"][0].task == "x"
        with pytest.raises(ValueError, match="no suite named"):
            suite_mod.get_suite("nope")
        with pytest.raises(ValueError, match="different suite"):
            suite_mod.register_suite(BenchmarkSuite(name="discovered", splits={"core": ("y",)}))
    finally:
        suite_mod._reset_registry_for_tests()


# ----------------------------------------------------------------------------------------------
# served policies
# ----------------------------------------------------------------------------------------------
class _DictObsEnv(_ScriptedEnv):
    """The scripted env with a tensor observation dict, so there is something to put on the wire."""

    def _obs(self):
        return {
            "robots": {"bot": {"joint_pos": torch.arange(self.num_envs * 2, dtype=torch.float32).reshape(-1, 2)}},
            "cameras": {"front": {"rgb": torch.zeros(self.num_envs, 4, 4, 3, dtype=torch.uint8)}},
        }

    def reset(self, states=None, env_ids=None, seed=None):
        super().reset(states=states, env_ids=env_ids, seed=seed)
        return self._obs(), {}

    def _observation(self, states):
        return self._obs()

    def step(self, actions):
        _, reward, terminated, time_out, info = super().step(actions)
        return self._obs(), reward, terminated, time_out, info


class _ChunkPolicy:
    """Returns a 3-step chunk ``default + 0.01 * k`` and records what it was given."""

    name = "chunker"

    def __init__(self):
        self.prompts, self.obs_keys, self.calls = [], None, 0

    def reset(self, env_ids, instances, ctx):
        self.prompts.extend(i.prompt for i in instances)

    def act(self, obs, ctx):
        self.calls += 1
        self.obs_keys = sorted(obs.robots["bot"].keys()), tuple(obs.cameras.front.rgb.shape)
        base = ctx.default_action.unsqueeze(0).repeat(ctx.num_envs, 1)
        return torch.stack([base + 0.01 * k for k in range(3)], dim=1)  # (num_envs, 3, action_dim)


@pytest.fixture
def served():
    servers = []

    def start(policy, **metadata):
        server = PolicyServer(policy, port=0, metadata=metadata)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield start
    for server, thread in servers:
        server.shutdown()
        thread.join(timeout=5)


def test_remote_hold_policy_scores_like_the_in_process_one(served):
    server = served(HoldPolicy(), checkpoint="none")
    local_env = _DictObsEnv(num_envs=2, schedule=[(3, 5, True), (None, 4, False)], pool_size=2)
    local = evaluate(local_env, HoldPolicy(), EvalProtocol(episodes=4, seed=7), task_name="t")
    remote_env = _DictObsEnv(num_envs=2, schedule=[(3, 5, True), (None, 4, False)], pool_size=2)
    remote_policy = RemotePolicy(server.url, required_metadata={"checkpoint": "none"})
    remote = evaluate(remote_env, remote_policy, EvalProtocol(episodes=4, seed=7), task_name="t")
    assert [(e.success, e.steps, e.instance_index) for e in remote.episodes] == [
        (e.success, e.steps, e.instance_index) for e in local.episodes
    ]
    assert all(torch.equal(a, b) for a, b in zip(local_env.actions, remote_env.actions, strict=True))


def test_remote_chunks_prompts_and_observation_paths(served):
    policy = _ChunkPolicy()
    server = served(policy)
    env = _DictObsEnv(num_envs=1, schedule=[(None, 7, False)], pool_size=1)
    env.prompt_bank = BANK
    result = evaluate(env, RemotePolicy(server.url), EvalProtocol(episodes=1, prompt_split="heldout"), task_name="t")
    assert result.episodes[0].steps == 7
    assert policy.calls == 3  # 7 steps from 3-step chunks
    offsets = [round(float(a[0, 0] - 0.5), 4) for a in env.actions]
    assert offsets == [0.0, 0.01, 0.02, 0.0, 0.01, 0.02, 0.0]
    assert policy.prompts == [result.episodes[0].prompt] and policy.prompts[0] in BANK.heldout
    assert policy.obs_keys == (["joint_pos"], (1, 4, 4, 3))


def test_remote_handshake_and_server_errors(served):
    server = served(HoldPolicy(), checkpoint="abc")
    with pytest.raises(ValueError, match="mismatch"):
        RemotePolicy(server.url, required_metadata={"checkpoint": "xyz"})

    class Broken(HoldPolicy):
        def act(self, obs, ctx):
            raise RuntimeError("model exploded")

    broken = served(Broken())
    env = _DictObsEnv(num_envs=1, schedule=[(None, 3, False)], pool_size=1)
    with pytest.raises(RuntimeError, match="model exploded"):
        evaluate(env, RemotePolicy(broken.url), EvalProtocol(episodes=1))
    with pytest.raises(ConnectionError):
        RemotePolicy("http://127.0.0.1:9", retries=0, timeout=2)


def test_observation_codec_roundtrip_and_key_filter():
    obs = {
        "robots": {"bot": {"joint_pos": torch.ones(2, 3), "name": "skip me"}},
        "cameras": {"front": {"rgb": np.zeros((2, 2, 2, 3), dtype=np.uint8), "depth": torch.zeros(2, 2, 2)}},
        "step": 4,
        "none": None,
    }
    flat = flatten_observation(obs)
    assert sorted(flat) == ["obs/cameras/front/depth", "obs/cameras/front/rgb", "obs/robots/bot/joint_pos", "obs/step"]
    tree = unflatten_observation(flat)
    assert torch.equal(tree.robots.bot.joint_pos, torch.ones(2, 3)) and tree.cameras["front"].rgb.dtype == torch.uint8
    only = flatten_observation(obs, keys=("robots", "cameras/front/rgb"))
    assert sorted(only) == ["obs/cameras/front/rgb", "obs/robots/bot/joint_pos"]


def test_actions_to_array_orders_dict_actions_by_joint():
    env = _ScriptedEnv(num_envs=2, schedule=[(None, 2, False)])
    from metasim.bench import build_context

    ctx = build_context(env, EvalProtocol())
    arr = actions_to_array([{"bot": {"dof_pos_target": {"j1": 1.5, "j0": -0.5}}}, {"bot": {"dof_pos_target": {}}}], ctx)
    assert arr.tolist() == [[-0.5, 1.5], [0.5, 1.0]]
