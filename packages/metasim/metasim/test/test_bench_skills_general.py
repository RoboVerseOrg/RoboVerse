"""Scripted skills, camera presets, the LLM agent's plumbing and the methods x tiers matrix, without a simulator."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.bench import BenchmarkResult, EpisodeResult, EvalProtocol, PolicyContext, matrix
from metasim.bench.cameras import CAMERA_PRESETS, camera_preset, with_cameras
from metasim.bench.skills import (
    ArmModel,
    Gripper,
    MoveJoints,
    MoveTo,
    ScriptedPolicy,
    SkillRunner,
    _rot_vec,
    env_view,
    object_yaw,
    plan_for,
    top_down_rot,
    wrap_symmetric,
)

pytestmark = pytest.mark.general

# a planar 3-link arm with a slide gripper, in the MJCF conventions the skills rely on: joints named
# like actuators, an end-effector body, finger joints that are not arm joints
_ARM_XML = """
<mujoco model="toy_arm">
  <worldbody>
    <body name="base">
      <body name="link1" pos="0 0 0.1">
        <joint name="arm_joint1" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="arm_joint2" axis="0 1 0" range="-3 3"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
          <body name="link3" pos="0.3 0 0">
            <joint name="arm_joint3" axis="0 1 0" range="-3 3"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
            <body name="hand" pos="0.2 0 0">
              <joint name="arm_joint4" axis="1 0 0" range="-3 3"/>
              <geom type="box" size="0.02 0.02 0.02"/>
              <body name="finger" pos="0 0.03 0">
                <joint name="finger_joint" type="slide" axis="0 1 0" range="0 0.04"/>
                <geom type="box" size="0.01 0.005 0.02"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _robot_cfg(tmp_path):
    path = tmp_path / "toy_arm.xml"
    path.write_text(_ARM_XML)
    return SimpleNamespace(
        name="toy",
        mjcf_path=str(path),
        ee_body_name="hand",
        ee_joint_names=["finger_joint"],
        actuators={
            "arm_joint1": None,
            "arm_joint2": None,
            "arm_joint3": None,
            "arm_joint4": None,
            "finger_joint": None,
        },
        joint_limits={
            "arm_joint1": (-3.0, 3.0),
            "arm_joint2": (-3.0, 3.0),
            "arm_joint3": (-3.0, 3.0),
            "arm_joint4": (-3.0, 3.0),
            "finger_joint": (0.0, 0.04),
        },
    )


def test_arm_model_fk_matches_the_mjcf_and_ik_reaches_a_pose(tmp_path):
    arm = ArmModel(_robot_cfg(tmp_path), tcp_offset=(0.05, 0.0, 0.0))
    assert arm.arm_joints == ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4"]
    assert arm.gripper_joints == ["finger_joint"] and arm.gripper_open == 0.04 and arm.gripper_closed == 0.0
    pos, rot = arm.fk(np.zeros(4))
    assert np.allclose(pos, [0.85, 0.0, 0.1]) and np.allclose(rot, np.eye(3)), "straight arm: links + tcp offset"
    target = np.array([0.5, 0.2, 0.3])
    q, pos_err, _ = arm.ik(np.zeros(4), target, None)
    assert pos_err < 1e-3, f"position-only IK did not converge ({pos_err})"
    assert np.allclose(arm.fk(q)[0], target, atol=1e-3)
    assert (q >= arm.low).all() and (q <= arm.high).all()


def test_ik_returns_the_closest_pose_when_the_target_is_out_of_reach(tmp_path):
    arm = ArmModel(_robot_cfg(tmp_path), tcp_offset=(0.0, 0.0, 0.0))
    q, pos_err, _ = arm.ik(np.zeros(4), np.array([2.0, 0.0, 0.1]), None, iters=60)
    assert 1.0 < pos_err < 1.3, "the arm is 0.8 m long; the solver stops at full extension"
    assert np.isfinite(q).all()


def test_rotation_helpers():
    assert np.allclose(_rot_vec(np.eye(3)), 0)
    rot = top_down_rot(0.0)
    assert np.allclose(rot[:, 2], [0, 0, -1]), "a top-down tool points at the table"
    assert abs(np.linalg.norm(_rot_vec(top_down_rot(0.3) @ top_down_rot(0.0).T)) - 0.3) < 1e-9
    quat = (math.cos(0.4), 0.0, 0.0, math.sin(0.4))  # 0.8 rad about z
    assert abs(object_yaw(quat) - 0.8) < 1e-9
    assert abs(wrap_symmetric(math.radians(100)) - math.radians(10)) < 1e-9, "a cube repeats every quarter turn"


def _obs(q: list[float], objects: dict[str, list[float]], robot: str = "toy"):
    def state(root):
        return SimpleNamespace(root_state=torch.tensor([root], dtype=torch.float32))

    rs = SimpleNamespace(
        joint_pos=torch.tensor([q], dtype=torch.float32),
        root_state=torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
    )
    return SimpleNamespace(
        robots={robot: rs},
        objects={k: state([*v, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) for k, v in objects.items()},
        cameras={},
    )


def test_env_view_reads_joints_and_object_poses_in_the_world():
    obs = _obs([0.0, 0.1, 0.2, 0.3, 0.4], {"cube": [0.5, 0.1, 0.02]})
    view = env_view(obs, 0, "toy", ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "finger_joint"])
    assert view.q["finger_joint"] == pytest.approx(0.4) and view.q["arm_joint2"] == pytest.approx(0.1)
    pos, quat = view.pose("cube")
    assert np.allclose(pos, [0.5, 0.1, 0.02]) and np.allclose(quat, [1, 0, 0, 0])
    with pytest.raises(KeyError, match="not observed"):
        view.pose("ghost")
    with pytest.raises(ValueError, match="no state for robot"):
        env_view(obs, 0, "other", [])


def test_skill_runner_moves_towards_the_target_and_logs_each_skill(tmp_path):
    arm = ArmModel(_robot_cfg(tmp_path), tcp_offset=(0.0, 0.0, 0.0))
    joints = [*arm.arm_joints, "finger_joint"]
    runner = SkillRunner(arm, "toy")
    goal = np.array([0.6, 0.1, 0.3])
    runner.load([MoveTo((goal, None), 0.04, speed=0.05, name="go"), Gripper(0.0, steps=2, name="close")])
    q = [0.0, 0.0, 0.0, 0.0, 0.0]
    first = runner.act(env_view(_obs(q, {}), 0, "toy", joints))["toy"]["dof_pos_target"]
    assert first["finger_joint"] == 0.04, "the move holds the gripper opening it was given"
    start = arm.fk(np.zeros(4))[0]
    after = arm.fk(np.array([first[j] for j in arm.arm_joints]))[0]
    assert np.linalg.norm(after - goal) < np.linalg.norm(start - goal), "the first waypoint brings the tool closer"
    assert np.linalg.norm(after - start) < 0.05 + 1e-3, "no further than `speed` per step"
    # drive the plan with perfect tracking until it finishes
    for _ in range(60):
        act = runner.act(env_view(_obs(q, {}), 0, "toy", joints))["toy"]["dof_pos_target"]
        q = [act[j] for j in joints]
        if runner.done:
            break
    assert runner.done and runner.log == ["go", "close"]
    assert np.allclose(arm.fk(np.array(q[:4]))[0], goal, atol=0.012)
    assert q[4] == 0.0, "the gripper closed"


def test_move_joints_interpolates_to_the_solved_configuration(tmp_path):
    arm = ArmModel(_robot_cfg(tmp_path), tcp_offset=(0.0, 0.0, 0.0))
    joints = [*arm.arm_joints, "finger_joint"]
    view = env_view(_obs([0.0] * 5, {}), 0, "toy", joints)
    skill = MoveJoints((np.array([0.4, 0.0, 0.5]), None), 0.0, steps=4)
    skill.start(view, arm)
    assert not skill.stalled
    q_mid, _, done = skill.step(view, arm)
    assert not done and np.allclose(q_mid, skill._q1 / 4, atol=1e-9), "one quarter of the way after one step"
    for _ in range(3):
        q_end, _, done = skill.step(view, arm)
    assert done and np.allclose(arm.fk(q_end)[0], [0.4, 0.0, 0.5], atol=2e-3)


def test_plan_lookup_accepts_registry_and_class_names():
    assert plan_for("pick_cube") is plan_for("PickCubeTask") is plan_for("maniskill.pick_cube")
    assert plan_for("PushCubeCfg") is plan_for("push_cube")
    with pytest.raises(KeyError, match="no scripted plan"):
        plan_for("open_drawer")


def test_scripted_policy_refuses_without_object_poses(tmp_path):
    cfg = _robot_cfg(tmp_path)
    ctx = PolicyContext(
        task="PickCubeTask",
        simulator="mujoco",
        robots=["toy"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=10,
        action_joint_names={"toy": sorted(cfg.actuators)},
        action_low=torch.zeros(5),
        action_high=torch.ones(5),
        default_action=torch.zeros(5),
        extras={"scenario": SimpleNamespace(robots=[cfg])},
    )
    policy = ScriptedPolicy()
    policy.reset([0], [None], ctx)
    with pytest.raises(KeyError, match="'cube' is not observed"):
        policy.act(_obs([0.0] * 5, {}), ctx)  # the visual tiers strip objects: the oracle cannot run there


def test_camera_presets_are_added_to_a_scenario_once():
    assert set(CAMERA_PRESETS) >= {"front", "side", "top"}
    assert camera_preset("front").name == "front" and camera_preset("front").data_types == ["rgb", "depth"]
    with pytest.raises(ValueError, match="unknown camera preset"):
        camera_preset("wrist")

    class _Scenario(SimpleNamespace):
        def replace(self, **kw):
            return _Scenario(**{**vars(self), **kw})

    scenario = _Scenario(cameras=[])
    out = with_cameras(scenario, ["front", "top"])
    assert [c.name for c in out.cameras] == ["front", "top"] and scenario.cameras == [], "the input is not mutated"
    with pytest.raises(ValueError, match="already has a camera named 'front'"):
        with_cameras(out, ["front"])
    assert with_cameras(scenario, []) is scenario


def _result(task, policy, wins, n, *, access=None, family=None):
    episodes = [
        EpisodeResult(
            index=i, env_id=0, instance_index=i, instance_id=f"{i}:0:x", seed=i, success=i < wins, steps=5,
            terminated=i < wins, time_out=i >= wins, return_sum=0.0,
        )
        for i in range(n)
    ]  # fmt: skip
    protocol = EvalProtocol(episodes=n).to_dict()
    if access is not None:
        protocol["access"] = access
    rec = {"name": policy}
    if family:
        rec["family"] = family
    return BenchmarkResult(
        task=task, simulator="mujoco", robots=["franka"], num_envs=1, protocol=protocol, policy=rec, episodes=episodes
    )


def test_matrix_crosses_methods_with_tiers_and_flags_partial_task_sets():
    visual = {"name": "visual", "observation": "rgb", "cameras": None, "proprioception": True, "tools": "none",
              "memory": "episode", "wall_time_s": None, "llm_calls": None, "llm_tokens": None, "tool_calls": None}  # fmt: skip
    results = [
        _result("pick", "scripted", 10, 10, family="scripted"),
        _result("push", "scripted", 8, 10, family="scripted"),
        _result("pick", "vla", 3, 10, access=visual, family="vla"),
        _result("pick", "hold", 0, 10),
        _result("push", "hold", 0, 10),
        _result("pick", "hold", 0, 10, access=visual),
        _result("push", "hold", 0, 10, access=visual),
    ]
    table = matrix(results)
    lines = table.splitlines()
    assert lines[0] == "| method | privileged | visual |", "privileged first, then the other tiers"
    rows = {line.split("|")[1].strip(): line for line in lines[2:6] if line.startswith("|")}
    assert (
        "0.900 (2 tasks, 20 ep)" in rows["scripted (scripted)"]
        and rows["scripted (scripted)"].rstrip("| ").endswith("ep) |") is False
    )
    assert rows["scripted (scripted)"].split("|")[3].strip() == "", "the oracle has no visual-tier cell"
    assert "0.300* (1 tasks, 10 ep)" in rows["vla (vla)"], "a method scored on fewer tasks is starred"
    assert "0.000 (2 tasks, 20 ep) | 0.000 (2 tasks, 20 ep)" in rows["hold"]
    assert "| pick | vla (vla) |  | 0.300 [0.11, 0.60] n=10 |" in table
    assert matrix([]) == "no results"
    json.dumps(table)  # plain text


def test_agent_builds_a_transcript_and_counts_usage_without_a_model(tmp_path, monkeypatch):
    """The agent's plumbing with a fake model: first turn content, tool dispatch, results, usage, truncation."""
    from metasim.bench import agent as agent_mod

    cfg = _robot_cfg(tmp_path)
    ctx = PolicyContext(
        task="PickCubeTask",
        simulator="mujoco",
        robots=["toy"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=50,
        action_joint_names={"toy": sorted(cfg.actuators)},
        action_low=torch.zeros(5),
        action_high=torch.ones(5),
        default_action=torch.zeros(5),
        instruction="Pick up the cube.",
        extras={"scenario": SimpleNamespace(robots=[cfg], cameras=[])},
        access={"name": "agent-privileged", "observation": "privileged", "tools": "sim_queries", "cameras": None},
    )
    replies = iter([
        [{"type": "text", "text": "query"}, {"type": "tool_use", "id": "t1", "name": "get_state", "input": {}}],
        [
            {"type": "tool_use", "id": "t2", "name": "grasp", "input": {}},
            {"type": "tool_use", "id": "t3", "name": "release", "input": {}},  # a second call in one turn is dropped
        ],
        [{"type": "tool_use", "id": "t4", "name": "done", "input": {"reason": "ok"}}],
    ])
    policy = agent_mod.LLMAgentPolicy(model="fake", max_calls=5, tcp_offset=(0.0, 0.0, 0.0))

    def fake_call(ep):
        policy._usage["llm_calls"] += 1
        policy._usage["llm_tokens"] += 100
        ep.calls += 1
        return next(replies)

    monkeypatch.setattr(policy, "_call", fake_call)
    joints = sorted(cfg.actuators)
    obs = _obs([0.0, 0.0, 0.0, 0.0, 0.04], {"cube": [0.5, 0.1, 0.02]})
    policy.reset([0], [SimpleNamespace(prompt=None)], ctx)
    act = policy.act(obs, ctx)[0]["toy"]["dof_pos_target"]
    ep = policy._episodes[0]
    first = ep.messages[0]["content"]
    assert (
        first[0]["text"] == "Instruction: Pick up the cube."
        and "cube: centre (0.500, 0.100, 0.020)" in first[1]["text"]
    )
    assert (
        ep.messages[1]["content"][-1]["name"] == "get_state" and ep.messages[2]["content"][0]["type"] == "tool_result"
    )
    assert [b["id"] for b in ep.messages[3]["content"] if b["type"] == "tool_use"] == ["t2"], "one tool per turn"
    assert act["finger_joint"] == 0.0, "the grasp tool closes the gripper"
    for _ in range(agent_mod.Gripper(0.0).steps + 1):
        policy.act(obs, ctx)
    assert ep.finished, "after the grasp the model said done"
    assert ep.messages[4]["content"][0]["content"].startswith("executed grasp: grasp.")
    assert policy.usage() == {"llm_calls": 3, "llm_tokens": 300, "tool_calls": 3, "llm_time_s": 0.0}
    assert ep.decisions[0].startswith("get_state({})") and ep.decisions[-1].startswith("done(")
    assert policy.act(obs, ctx)[0]["toy"]["dof_pos_target"][joints[0]] == pytest.approx(0.0), "holds once done"


def test_agent_refuses_a_tier_without_tools(tmp_path):
    from metasim.bench.agent import LLMAgentPolicy

    cfg = _robot_cfg(tmp_path)
    ctx = PolicyContext(
        task="t", simulator="mujoco", robots=["toy"], num_envs=1, device=torch.device("cpu"), seed=0,
        max_episode_steps=5, action_joint_names={"toy": sorted(cfg.actuators)}, action_low=torch.zeros(5),
        action_high=torch.ones(5), default_action=torch.zeros(5), extras={"scenario": SimpleNamespace(robots=[cfg])},
        access={"name": "visual", "observation": "rgb", "tools": "none"},
    )  # fmt: skip
    with pytest.raises(ValueError, match="allows no tools"):
        LLMAgentPolicy().reset([0], [None], ctx)
