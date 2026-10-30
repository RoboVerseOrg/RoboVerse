"""The learning stacks' policy adapters satisfy ``metasim.bench.Policy`` and reproduce their evaluators' actions."""

from __future__ import annotations

import pytest
import torch

from metasim.bench import Policy, PolicyContext, load_policy


def _ctx(num_envs: int, action_dim: int) -> PolicyContext:
    return PolicyContext(
        task="t",
        simulator="mujoco",
        robots=["franka"],
        num_envs=num_envs,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=10,
        action_joint_names={"franka": [f"j{i}" for i in range(action_dim)]},
        action_low=-torch.ones(action_dim),
        action_high=torch.ones(action_dim),
        default_action=torch.zeros(action_dim),
    )


def _needs_fast_td3():
    """FastTD3 lives behind the ``[learn]`` extra: its module imports ``tensordict``."""
    pytest.importorskip("tensordict", reason="FastTD3 needs tensordict (pip install -e '.[learn]')")


def _fast_td3_checkpoint(tmp_path, n_obs: int, n_act: int, hidden: int):
    _needs_fast_td3()
    from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization

    torch.manual_seed(3)
    actor = Actor(n_obs=n_obs, n_act=n_act, num_envs=4, init_scale=0.1, hidden_dim=hidden, device=torch.device("cpu"))
    normalizer = EmpiricalNormalization(shape=n_obs, device=torch.device("cpu"))
    normalizer.train()
    normalizer(torch.randn(64, n_obs) * 3 + 1)
    normalizer.eval()
    path = tmp_path / "final.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_normalizer_state": normalizer.state_dict(),
            "config": {"task": "t", "actor_hidden_dim": hidden, "init_scale": 0.1},
            "global_step": 7,
        },
        path,
    )
    return path, actor, normalizer


def test_fast_td3_policy_reproduces_the_evaluator_actions(tmp_path):
    path, actor, normalizer = _fast_td3_checkpoint(tmp_path, n_obs=6, n_act=3, hidden=32)
    policy = load_policy("roboverse_learn.rl.fast_td3.bench_policy:FastTD3Policy", checkpoint=str(path))
    assert isinstance(policy, Policy) and policy.name == "fast_td3"
    ctx = _ctx(num_envs=2, action_dim=3)  # a different num_envs than the checkpoint trained with
    policy.reset([0, 1], [], ctx)
    obs = torch.randn(2, 6)
    actions = policy.act(obs, ctx)
    with torch.no_grad():
        expected = actor(normalizer(obs)).float()
    assert actions.shape == (2, 3)
    assert torch.allclose(actions, expected, atol=1e-6)
    assert policy.config["task"] == "t"


def test_fast_td3_policy_rejects_a_checkpoint_for_another_task(tmp_path):
    path, _, _ = _fast_td3_checkpoint(tmp_path, n_obs=6, n_act=3, hidden=32)
    from roboverse_learn.rl.fast_td3.bench_policy import FastTD3Policy

    policy = FastTD3Policy(checkpoint=str(path))
    with pytest.raises(ValueError, match="does not fit this task"):
        policy.act(torch.randn(1, 6), _ctx(num_envs=1, action_dim=5))  # action width differs
    policy = FastTD3Policy(checkpoint=str(path))
    with pytest.raises(ValueError, match="does not fit this task"):
        policy.act(torch.randn(1, 9), _ctx(num_envs=1, action_dim=3))  # observation width differs


def test_fast_td3_policy_needs_a_fast_td3_checkpoint(tmp_path):
    _needs_fast_td3()
    from roboverse_learn.rl.fast_td3.bench_policy import FastTD3Policy

    path = tmp_path / "other.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="no 'actor_state_dict'"):
        FastTD3Policy(checkpoint=str(path))


class _FakeRunner:
    def __init__(self):
        self.resets = 0
        self.seen = []

    def reset(self):
        self.resets += 1

    def get_action(self, obs):
        self.seen.append(obs)
        return [{"franka": {"dof_pos_target": {"j0": float(obs["joint_qpos"][i, 0])}}} for i in range(len(obs["rgb"]))]


class _Cam:
    def __init__(self, rgb):
        self.rgb = rgb


class _Rob:
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos


class _Obs:
    def __init__(self, cameras, robots):
        self.cameras = cameras
        self.robots = robots


def test_il_policy_feeds_the_runner_the_evaluators_observation():
    from roboverse_learn.il.runners.bench_policy import ILPolicy

    runner = _FakeRunner()
    policy = ILPolicy(runner=runner, camera="camera0")
    assert isinstance(policy, Policy)
    ctx = _ctx(num_envs=2, action_dim=1)
    policy.reset([0, 1], [], ctx)
    rgb = torch.zeros(2, 4, 4, 3)
    qpos = torch.tensor([[0.25], [0.75]])
    actions = policy.act(_Obs({"camera0": _Cam(rgb)}, {"franka": _Rob(qpos)}), ctx)
    assert runner.resets == 1
    assert runner.seen[0]["rgb"] is rgb and runner.seen[0]["joint_qpos"] is qpos
    assert actions[1]["franka"]["dof_pos_target"]["j0"] == 0.75


def test_il_policy_names_the_missing_camera():
    from roboverse_learn.il.runners.bench_policy import ILPolicy

    policy = ILPolicy(runner=_FakeRunner(), camera="head")
    ctx = _ctx(num_envs=1, action_dim=1)
    policy.reset([0], [], ctx)
    with pytest.raises(KeyError, match="needs camera 'head'"):
        policy.act(_Obs({"camera0": _Cam(torch.zeros(1, 2, 2, 3))}, {"franka": _Rob(torch.zeros(1, 1))}), ctx)


def test_il_policy_requires_a_runner_or_checkpoint():
    from roboverse_learn.il.runners.bench_policy import ILPolicy

    with pytest.raises(ValueError, match="runner or a checkpoint"):
        ILPolicy()
