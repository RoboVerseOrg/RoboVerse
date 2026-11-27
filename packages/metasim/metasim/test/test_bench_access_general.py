"""General (non-GPU) tests for the information-access tiers (``metasim.bench.access``)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import torch

from metasim.bench import (
    ACCESS_PROFILES,
    AccessProfile,
    EvalProtocol,
    PolicyContext,
    evaluate,
    get_access_profile,
    restrict_observation,
)
from metasim.bench.access import usage_report
from metasim.bench.remote import _context_from_json, _context_to_json
from metasim.bench.result import BenchmarkResult, EpisodeResult, result_key, summarize
from metasim.bench.suite import BenchmarkSuite, SuiteTask, _protocol_mismatch, suite_report, suite_table
from metasim.test.test_bench_general import _ScriptedEnv
from metasim.types import CameraState, ObjectState, RobotState, TensorState


def _tensor_state(num_envs: int = 2, cameras: tuple[str, ...] = ("front", "wrist")) -> TensorState:
    root = torch.zeros(num_envs, 13)
    cams = {
        name: CameraState(
            rgb=torch.zeros(num_envs, 4, 4, 3, dtype=torch.uint8),
            depth=torch.ones(num_envs, 4, 4),
            instance_seg=torch.zeros(num_envs, 4, 4, dtype=torch.int32),
            pos=torch.zeros(num_envs, 3),
            quat_world=torch.tensor([[1.0, 0, 0, 0]] * num_envs),
        )
        for name in cameras
    }
    return TensorState(
        objects={"cube": ObjectState(root_state=root.clone())},
        robots={"franka": RobotState(root_state=root.clone(), joint_pos=torch.zeros(num_envs, 9))},
        cameras=cams,
        extras={"contact": torch.zeros(num_envs, 1)},
    )


@pytest.mark.general
def test_presets_are_valid_and_the_default_is_unrestricted():
    assert AccessProfile() == ACCESS_PROFILES["privileged"]
    assert not AccessProfile().restricts_observation
    for name, profile in ACCESS_PROFILES.items():
        assert profile.name == name
        assert AccessProfile.from_dict(profile.to_dict()) == profile
    assert get_access_profile(None) == AccessProfile()
    assert get_access_profile("visual").observation == "rgb"
    assert get_access_profile(ACCESS_PROFILES["visual"]) is ACCESS_PROFILES["visual"]


@pytest.mark.general
def test_profile_validation():
    with pytest.raises(ValueError, match="observation"):
        AccessProfile(observation="lidar")
    with pytest.raises(ValueError, match="tools"):
        AccessProfile(tools="telekinesis")
    with pytest.raises(ValueError, match="memory"):
        AccessProfile(memory="forever")
    with pytest.raises(ValueError, match="cameras"):
        AccessProfile(cameras=())
    with pytest.raises(ValueError, match="wall_time_s"):
        AccessProfile(wall_time_s=0)
    with pytest.raises(ValueError, match="llm_tokens"):
        AccessProfile(llm_tokens=-1)
    with pytest.raises(ValueError, match="unknown access profile"):
        get_access_profile("nope")


@pytest.mark.general
def test_profile_from_a_json_file(tmp_path):
    path = tmp_path / "tier.json"
    path.write_text(json.dumps({"name": "lab", "observation": "rgb", "cameras": ["front"], "llm_calls": 20}))
    profile = get_access_profile(str(path))
    assert profile.cameras == ("front",) and profile.budgets == {"llm_calls": 20}


@pytest.mark.general
def test_privileged_observation_passes_through_untouched():
    obs = _tensor_state()
    assert restrict_observation(obs, AccessProfile()) is obs


@pytest.mark.general
def test_rgb_tier_removes_objects_extras_depth_and_masks_but_keeps_proprioception_and_rgb():
    obs = _tensor_state()
    seen = restrict_observation(obs, ACCESS_PROFILES["visual"])
    assert seen is not obs
    assert seen.objects == {} and seen.extras == {}
    assert set(seen.robots) == {"franka"} and torch.equal(
        seen.robots["franka"].joint_pos, obs.robots["franka"].joint_pos
    )
    assert set(seen.cameras) == {"front", "wrist"}
    cam = seen.cameras["front"]
    assert cam.rgb is not None and cam.depth is None and cam.instance_seg is None and cam.pos is not None
    # the original is untouched
    assert obs.cameras["front"].depth is not None and obs.objects


@pytest.mark.general
def test_depth_tier_keeps_depth_and_camera_subset_is_enforced():
    obs = _tensor_state()
    seen = restrict_observation(obs, AccessProfile(name="d", observation="rgb_depth", cameras=("front",)))
    assert set(seen.cameras) == {"front"} and seen.cameras["front"].depth is not None
    assert seen.cameras["front"].instance_seg is None
    with pytest.raises(ValueError, match="does not render"):
        restrict_observation(obs, AccessProfile(name="x", observation="rgb", cameras=("overhead",)))


@pytest.mark.general
def test_no_proprioception_drops_the_robots_even_under_privileged_objects():
    obs = _tensor_state()
    seen = restrict_observation(obs, AccessProfile(name="blind", proprioception=False))
    assert seen.robots == {} and seen.objects  # privileged observation keeps the objects
    assert set(seen.cameras) == {"front", "wrist"}


@pytest.mark.general
def test_a_non_tensor_observation_cannot_be_restricted():
    with pytest.raises(TypeError, match="cannot be enforced"):
        restrict_observation(torch.zeros(2, 7), ACCESS_PROFILES["visual"])
    assert restrict_observation(torch.zeros(2, 7), AccessProfile()) is not None  # privileged: pass-through


@pytest.mark.general
def test_protocol_carries_the_tier_and_round_trips():
    protocol = EvalProtocol(episodes=3, access=ACCESS_PROFILES["agent-skills"])
    data = protocol.to_dict()
    assert data["access"]["name"] == "agent-skills" and data["access"]["tools"] == "skills"
    assert EvalProtocol.from_dict(data) == protocol
    assert EvalProtocol.from_dict(json.loads(json.dumps(data))) == protocol
    # an old result without the field reads as privileged
    old = {k: v for k, v in data.items() if k != "access"}
    assert EvalProtocol.from_dict(old).access == AccessProfile()
    with pytest.raises(ValueError, match="access"):
        EvalProtocol(access="visual")


def _result(access: AccessProfile, success: bool = True) -> BenchmarkResult:
    return BenchmarkResult(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=1,
        protocol=EvalProtocol(episodes=1, access=access).to_dict(),
        policy={"name": "p"},
        episodes=[
            EpisodeResult(
                index=0,
                env_id=0,
                instance_index=0,
                instance_id="a",
                seed=0,
                success=success,
                steps=5,
                terminated=True,
                time_out=False,
                return_sum=1.0,
                wall_time_s=0.5,
            )
        ],
    )


@pytest.mark.general
def test_results_under_different_tiers_are_not_pooled_and_the_table_shows_the_tier():
    privileged, visual = _result(AccessProfile()), _result(ACCESS_PROFILES["visual"], success=False)
    assert result_key(privileged) != result_key(visual)
    table = summarize([privileged, visual])
    rows = [ln for ln in table.splitlines() if ln.startswith("| t |")]
    assert len(rows) == 2 and any("| visual |" in r for r in rows) and any("| privileged |" in r for r in rows)
    back = BenchmarkResult.from_dict(json.loads(json.dumps(visual.to_dict())))
    assert back.episodes[0].wall_time_s == 0.5 and back.episodes[0].over_budget is None


@pytest.mark.general
def test_suite_rejects_a_result_from_another_tier_and_reports_dimensions():
    suite = BenchmarkSuite(
        name="s",
        splits={"core": (SuiteTask("t", dimensions=("precision", "pick_place")),)},
        protocol=EvalProtocol(episodes=1, access=ACCESS_PROFILES["visual"]),
    )
    mismatch = _protocol_mismatch(_result(AccessProfile()), suite.protocol_for(suite.entries("core")[0][1]))
    assert any(m.startswith("access=") for m in mismatch)
    report = suite_report(suite, [_result(ACCESS_PROFILES["visual"])])
    block = report["splits"]["core"]
    assert block["complete"] and block["dimensions"]["precision"]["macro_success_rate"] == 1.0
    assert block["tasks"][0]["dimensions"] == ["precision", "pick_place"]
    assert "core by dimension: pick_place 1.000, precision 1.000" in suite_table(report)


@pytest.mark.general
def test_context_json_carries_the_tier():
    ctx = PolicyContext(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=10,
        action_joint_names={"franka": ["j0"]},
        action_low=torch.zeros(1),
        action_high=torch.ones(1),
        default_action=torch.zeros(1),
        access=ACCESS_PROFILES["visual"].to_dict(),
    )
    back = _context_from_json(_context_to_json(ctx), "cpu")
    assert back.access["name"] == "visual" and back.access["memory"] == "episode"
    assert _context_from_json(_context_to_json(PolicyContext(**{**ctx.__dict__, "access": None})), "cpu").access is None


@pytest.mark.general
def test_usage_report_compares_per_episode_means_with_the_budgets():
    access = AccessProfile(name="a", llm_calls=10, llm_tokens=1000)
    report = usage_report({"llm_calls": 50, "llm_tokens": 2000, "extra": 1}, access, episodes=4)
    assert report["per_episode"] == {"llm_calls": 12.5, "llm_tokens": 500.0}
    assert report["exceeded"] == ["llm_calls"] and "unreported" not in report
    assert usage_report({"llm_calls": 3}, access, episodes=1)["unreported"] == ["llm_tokens"]
    assert usage_report(None, AccessProfile(), episodes=3) is None
    assert usage_report(None, access, episodes=3)["unreported"] == ["llm_calls", "llm_tokens"]


# --- the evaluator under a tier -------------------------------------------------------------------------


class _ImageEnv(_ScriptedEnv):
    """The scripted env observing a ``TensorState`` with cameras, so a tier below privileged can be enforced."""

    def __init__(self, num_envs, schedule, pool_size=0, reset_sleep=0.0, cameras=("front", "wrist")):
        super().__init__(num_envs, schedule, pool_size)
        self.scenario = SimpleNamespace(simulator="fake", robots=self.scenario.robots, objects=["cube"])
        self.reset_sleep = reset_sleep
        self.cameras = cameras

    def _obs(self):
        return _tensor_state(self.num_envs, cameras=self.cameras)

    def reset(self, states=None, env_ids=None, seed=None):
        super().reset(states, env_ids, seed)
        time.sleep(self.reset_sleep)
        return self._obs(), {}

    def _observation(self, states):
        return self._obs()

    def step(self, actions):
        _, reward, terminated, time_out, info = super().step(actions)
        return self._obs(), reward, terminated, time_out, info


class _Spy:
    """Records what ``reset`` and ``act`` are given; optionally slow, optionally reporting usage."""

    name = "spy"

    def __init__(self, act_sleep=0.0, usage=None):
        self.resets: list[list] = []
        self.ctx = None
        self.seen = []
        self.act_sleep = act_sleep
        self._usage = usage

    def reset(self, env_ids, instances, ctx):
        self.resets.append(list(instances))
        self.ctx = ctx

    def act(self, obs, ctx):
        time.sleep(self.act_sleep)
        self.seen.append(obs)
        return torch.zeros(ctx.num_envs, 2)

    def usage(self):
        return self._usage


@pytest.mark.general
def test_a_restricted_tier_hides_the_instance_state_and_the_object_list_at_reset():
    for access, hidden in [(ACCESS_PROFILES["visual"], True), (AccessProfile(), False)]:
        env, policy = _ImageEnv(1, [(1, 2, False)], pool_size=2), _Spy()
        result = evaluate(env, policy, EvalProtocol(episodes=2, access=access))
        assert [e.success for e in result.episodes] == [True, True]
        instances = [inst for batch in policy.resets for inst in batch]
        assert len(instances) == 2
        assert all((inst.state is None) is hidden for inst in instances)
        assert all((inst.actions is None) is hidden or inst.actions is None for inst in instances)
        assert (policy.ctx.extras["scenario"].objects == []) is hidden
        # the env itself was still reset from the explicit instance states
        assert all(states is not None for _, states, _ in env.resets)
        if hidden:
            assert all(obs.objects == {} and obs.cameras["front"].depth is None for obs in policy.seen)


@pytest.mark.general
def test_wall_clock_budget_starts_after_the_reset_and_ends_the_episode_as_a_time_out():
    budget = AccessProfile(name="clock", wall_time_s=0.1)
    slow_reset = evaluate(
        _ImageEnv(1, [(1, 2, False)], reset_sleep=0.25), _Spy(), EvalProtocol(episodes=1, access=budget)
    )
    (episode,) = slow_reset.episodes
    assert episode.over_budget is None and episode.steps == 2 and episode.wall_time_s < 0.1
    assert slow_reset.summary["over_budget"] == 0

    slow_policy = evaluate(_ImageEnv(1, [(1, 5, False)]), _Spy(act_sleep=0.15), EvalProtocol(episodes=1, access=budget))
    (episode,) = slow_policy.episodes
    # the budget ends the episode at the first step past it; the success rule still judges the steps taken
    assert episode.over_budget is True and episode.time_out is True and episode.steps == 1 and episode.success is True
    assert slow_policy.summary["over_budget"] == 1 and slow_policy.notes["access"] == "clock"
    back = BenchmarkResult.from_dict(json.loads(json.dumps(slow_policy.to_dict())))
    assert back.episodes[0].over_budget is True and back.summary["over_budget"] == 1


@pytest.mark.general
def test_usage_keeps_finite_numbers_only():
    usage = {"llm_calls": "n/a", "llm_tokens": torch.tensor(3), "tool_calls": 2.0, "nan": float("nan"), "ok": True}
    access = AccessProfile(name="budgeted", llm_calls=1, llm_tokens=10)
    result = evaluate(_ImageEnv(1, [(1, 2, False)]), _Spy(usage=usage), EvalProtocol(episodes=1, access=access))
    assert result.policy["usage"] == {"llm_tokens": 3, "tool_calls": 2.0, "ok": 1}
    assert result.notes["budget"]["exceeded"] == [] and result.notes["budget"]["unreported"] == ["llm_calls"]
    assert result.save  # the record is JSON-serialisable
    json.dumps(result.to_dict())


@pytest.mark.general
def test_tiers_are_compared_by_the_whole_profile_not_the_name():
    relabelled = AccessProfile(name="visual", observation="privileged", tools="sim_queries")
    assert result_key(_result(relabelled)) != result_key(_result(ACCESS_PROFILES["visual"]))
    assert result_key(_result(AccessProfile())) == result_key(_result(ACCESS_PROFILES["privileged"]))
    old = _result(AccessProfile())
    old.protocol.pop("access")  # a result written before tiers existed is the privileged tier
    assert result_key(old) == result_key(_result(AccessProfile()))


@pytest.mark.general
def test_profile_files_need_a_name_and_cannot_shadow_a_preset(tmp_path):
    nameless = tmp_path / "nameless.json"
    nameless.write_text(json.dumps({"observation": "rgb"}))
    with pytest.raises(ValueError, match="name"):
        get_access_profile(str(nameless))
    shadow = tmp_path / "shadow.json"
    shadow.write_text(json.dumps({"name": "visual", "observation": "privileged"}))
    with pytest.raises(ValueError, match="preset"):
        get_access_profile(str(shadow))
    same = tmp_path / "same.json"
    same.write_text(json.dumps(ACCESS_PROFILES["visual"].to_dict()))
    assert get_access_profile(str(same)) == ACCESS_PROFILES["visual"]
    with pytest.raises(ValueError, match="unknown fields"):
        AccessProfile.from_dict({"name": "x", "observations": "rgb"})
    with pytest.raises(ValueError, match="cameras must be a list"):
        AccessProfile.from_dict({"name": "x", "cameras": "front"})


@pytest.mark.general
def test_an_image_tier_on_a_task_without_cameras_is_an_error():
    with pytest.raises(ValueError, match="renders no camera"):
        restrict_observation(_tensor_state(cameras=()), ACCESS_PROFILES["visual"])
    assert restrict_observation(_tensor_state(cameras=()), AccessProfile()).cameras == {}
    restricted = restrict_observation(_tensor_state(), ACCESS_PROFILES["visual"])
    assert restricted.cameras["front"].instance_seg_id2label is None


@pytest.mark.general
def test_a_suite_run_under_a_tier_is_scored_under_that_tier():
    from metasim.__main__ import _suite_under_access

    suite = BenchmarkSuite(name="s", splits={"core": ("t",)}, protocol=EvalProtocol(episodes=1))
    assert _suite_under_access(suite, None) is suite
    tiered = _suite_under_access(suite, "visual")
    assert tiered.protocol_for(tiered.entries("core")[0][1]).access == ACCESS_PROFILES["visual"]
    assert suite_report(tiered, [_result(ACCESS_PROFILES["visual"])])["splits"]["core"]["complete"]
    assert not suite_report(suite, [_result(ACCESS_PROFILES["visual"])])["splits"]["core"]["complete"]
    with pytest.raises(SystemExit, match="invalid --access"):
        _suite_under_access(suite, "no-such-tier")
