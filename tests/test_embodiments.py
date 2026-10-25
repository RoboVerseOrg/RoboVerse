"""The reference embodiments (Franka, YAM, Sharpa Wave) load, hold and drive on MuJoCo, and run under the bench.

Every robot config here is one the benchmark tables report by (``ROADMAP.md`` §4). The tests pin the contract
each config promises: it resolves by name, its joints are the asset's joints, its default pose holds, every
actuated joint tracks a target, and the benchmark harness evaluates a policy on it. Asset-backed tests skip
with the fetch instructions when ``roboverse_data`` lacks the files (``requires_asset``).
"""

from __future__ import annotations

import pytest

from roboverse_pack.robots import FrankaCfg, SharpaWaveLeftCfg, SharpaWaveRightCfg, YamCfg

YAM_MJCF = "robots/Yet_Another_Manipulator_YAM/mjcf/yam.xml"
SHARPA_RIGHT_MJCF = "robots/sharpa_wave/mjcf/right_hand.xml"
SHARPA_LEFT_MJCF = "robots/sharpa_wave/mjcf/left_hand.xml"
FRANKA_MJCF = "robots/franka/mjcf/panda.xml"

_EMBODIMENTS = [
    pytest.param(FrankaCfg, FRANKA_MJCF, id="franka"),
    pytest.param(YamCfg, YAM_MJCF, id="yam"),
    pytest.param(SharpaWaveRightCfg, SHARPA_RIGHT_MJCF, id="sharpa_wave_right"),
    pytest.param(SharpaWaveLeftCfg, SHARPA_LEFT_MJCF, id="sharpa_wave_left"),
]


def _mujoco_handler(robot_cfg, num_envs: int = 1):
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.sim_context import HandlerContext

    return HandlerContext(ScenarioCfg(simulator="mujoco", num_envs=num_envs, headless=True, robots=[robot_cfg]))


def _hold(handler, name: str, steps: int):
    target = handler.get_states(mode="tensor").robots[name].joint_pos.clone()
    for _ in range(steps):
        handler.set_dof_targets(target)
        handler.simulate()
    return target, handler.get_states(mode="tensor").robots[name]


@pytest.mark.parametrize("name", ["franka", "yam", "sharpa_wave_right", "sharpa_wave_left"])
def test_embodiment_resolves_by_name(name):
    from metasim.utils.setup_util import get_robot

    cfg = get_robot(name)
    assert cfg.name == name
    assert cfg.num_joints == len(cfg.joint_limits), "joint_limits enumerates every joint of the asset"
    assert set(cfg.default_joint_positions) == set(cfg.joint_limits)
    for joint, (lo, hi) in cfg.joint_limits.items():
        assert lo <= cfg.default_joint_positions[joint] <= hi, joint


def test_yam_config_matches_its_asset_conventions():
    cfg = YamCfg()
    assert set(cfg.actuators) == {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "left_finger"}
    assert "right_finger" in cfg.joint_limits and "right_finger" not in cfg.actuators, "the coupled finger is passive"
    assert cfg.default_joint_positions["right_finger"] == -cfg.default_joint_positions["left_finger"]
    assert cfg.gripper_open_q == [cfg.joint_limits["left_finger"][1]]
    assert cfg.file_type["newton"] == "mjcf"


def test_sharpa_wave_hands_mirror_each_other():
    right, left = SharpaWaveRightCfg(), SharpaWaveLeftCfg()
    strip = lambda d, side: {k[len(side) + 1 :]: v for k, v in d.items()}  # noqa: E731
    assert strip(right.joint_limits, "right") == strip(left.joint_limits, "left")
    assert {k: (a.stiffness, a.damping, a.effort_limit_sim) for k, a in strip(right.actuators, "right").items()} == {
        k: (a.stiffness, a.damping, a.effort_limit_sim) for k, a in strip(left.actuators, "left").items()
    }
    assert len(right.actuators) == 22


def _skip_without_asset(relpath: str) -> None:
    from conftest import missing_assets

    if missing_assets((relpath,)):
        pytest.skip(f"roboverse_data asset not fetched: {relpath}")


@pytest.mark.parametrize(("cfg_cls", "asset"), _EMBODIMENTS)
def test_embodiment_loads_and_holds_on_mujoco(cfg_cls, asset):
    """Joint vector = the asset's joints; the default pose holds; every actuated joint tracks a target."""
    _skip_without_asset(asset)
    cfg = cfg_cls()
    with _mujoco_handler(cfg) as handler:
        names = handler.get_joint_names(cfg.name, sort=True)
        assert names == sorted(cfg.joint_limits), "config joints are exactly the asset joints"
        assert handler._passive_joint_names(cfg.name) == sorted(set(cfg.joint_limits) - set(cfg.actuators))

        target, state = _hold(handler, cfg.name, steps=50)
        assert state.joint_pos.shape == (1, cfg.num_joints)
        assert state.joint_pos_target.shape == (1, cfg.num_joints)
        assert float((state.joint_pos - target).abs().max()) < 0.02, "the default pose holds against gravity"

        # Drive the first actuated joint a quarter of the way through its range and expect it to track.
        joint = next(j for j in names if j in cfg.actuators)
        lo, hi = cfg.joint_limits[joint]
        col = names.index(joint)
        target[0, col] = lo + 0.25 * (hi - lo)
        for _ in range(300):
            handler.set_dof_targets(target)
            handler.simulate()
        state = handler.get_states(mode="tensor").robots[cfg.name]
        assert abs(float(state.joint_pos[0, col]) - float(target[0, col])) < 0.05 * (hi - lo) + 0.01, joint
        assert float(state.joint_pos_target[0, col]) == pytest.approx(float(target[0, col]))


def test_yam_gripper_closes_symmetrically_on_mujoco():
    """Commanding ``left_finger`` moves both fingers: the passive finger follows through the equality constraint."""
    _skip_without_asset(YAM_MJCF)
    cfg = YamCfg()
    with _mujoco_handler(cfg) as handler:
        names = handler.get_joint_names("yam", sort=True)
        left, right = names.index("left_finger"), names.index("right_finger")
        target, _ = _hold(handler, "yam", steps=20)
        target[0, left] = cfg.gripper_close_q[0]
        target[0, right] = 123.0  # a passive joint ignores whatever its action slot carries
        for _ in range(150):
            handler.set_dof_targets(target)
            handler.simulate()
        state = handler.get_states(mode="tensor").robots["yam"]
        assert float(state.joint_pos[0, left]) == pytest.approx(cfg.gripper_close_q[0], abs=2e-3)
        assert float(state.joint_pos[0, right]) == pytest.approx(-float(state.joint_pos[0, left]), abs=1e-3)
        assert float(state.joint_pos_target[0, right]) == pytest.approx(float(state.joint_pos[0, right]))
        assert float(state.joint_pos_target[0, left]) == pytest.approx(cfg.gripper_close_q[0])


def test_yam_runs_under_the_bench_harness():
    """A plain task on YAM evaluates with the hold baseline and produces a valid result (success undefined)."""
    _skip_without_asset(YAM_MJCF)
    from metasim.bench import EvalProtocol, HoldPolicy, evaluate, validate_result
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.task.base import BaseTaskEnv

    class HoldYam(BaseTaskEnv):
        max_episode_steps = 4

        def _get_initial_states(self):
            return None

    scenario = ScenarioCfg(simulator="mujoco", num_envs=1, headless=True, robots=[YamCfg()])
    env = HoldYam(scenario, device="cpu")
    try:
        result = evaluate(env, HoldPolicy(), EvalProtocol(episodes=2, max_episode_steps=4), task_name="hold_yam")
    finally:
        env.close()
    validate_result(result.to_dict())
    assert result.robots == ["yam"]
    assert len(result.episodes) == 2
    assert result.summary["success_rate"] == 0.0, "a BaseTaskEnv scores success as termination; hold never terminates"
    assert all(episode.steps == 4 for episode in result.episodes)


def test_yam_drives_through_the_parallel_wrapper():
    """Two MuJoCo envs: joint names, passive-joint handling and targets all travel through the worker pipes."""
    _skip_without_asset(YAM_MJCF)
    cfg = YamCfg()
    with _mujoco_handler(cfg, num_envs=2) as handler:
        names = handler.get_joint_names("yam", sort=True)
        assert names == sorted(cfg.joint_limits)
        state = handler.get_states(mode="tensor").robots["yam"]
        assert state.joint_pos.shape == (2, cfg.num_joints)
        target = state.joint_pos.clone()
        target[1, names.index("joint1")] = 0.5  # move env 1 only
        for _ in range(200):
            handler.set_dof_targets(target)
            handler.simulate()
        state = handler.get_states(mode="tensor").robots["yam"]
        assert float(state.joint_pos[1, names.index("joint1")]) == pytest.approx(0.5, abs=0.02)
        assert float(state.joint_pos[0, names.index("joint1")]) == pytest.approx(0.0, abs=0.02)
        assert state.joint_pos_target.shape == (2, cfg.num_joints)
