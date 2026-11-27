"""The end-effector-delta VLA adapter, with a stand-in model: deltas become joint targets through the kinematics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.bench import PolicyContext
from metasim.bench.skills import ArmModel
from roboverse_learn.vla.bench_policy import EEDeltaPolicy

_ARM_XML = """
<mujoco model="toy_arm">
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="arm_joint1" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="arm_joint2" axis="0 1 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        <body name="hand" pos="0.3 0 0">
          <joint name="arm_joint3" axis="0 1 0" range="-3 3"/>
          <geom type="box" size="0.02 0.02 0.02"/>
          <body name="finger" pos="0 0.03 0">
            <joint name="finger_joint" type="slide" axis="0 1 0" range="0 0.04"/>
            <geom type="box" size="0.01 0.005 0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _ConstantVLA(EEDeltaPolicy):
    """Always asks for +2 cm in z and an open gripper; records what it was shown."""

    name = "constant"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list[tuple[tuple[int, ...], str]] = []

    def predict(self, image, instruction):
        self.seen.append((tuple(image.shape), instruction))
        return np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0])


def _ctx_and_obs(tmp_path, monkeypatch):
    path = tmp_path / "toy.xml"
    path.write_text(_ARM_XML)
    cfg = SimpleNamespace(
        name="toy",
        mjcf_path=str(path),
        ee_body_name="hand",
        ee_joint_names=["finger_joint"],
        actuators={"arm_joint1": None, "arm_joint2": None, "arm_joint3": None, "finger_joint": None},
        joint_limits={
            "arm_joint1": (-3.0, 3.0),
            "arm_joint2": (-3.0, 3.0),
            "arm_joint3": (-3.0, 3.0),
            "finger_joint": (0.0, 0.04),
        },
    )
    import metasim.utils.setup_util as setup_util

    monkeypatch.setattr(setup_util, "get_robot", lambda name: cfg)
    joints = sorted(cfg.actuators)
    ctx = PolicyContext(
        task="PickCubeTask",
        simulator="mujoco",
        robots=["toy"],
        num_envs=1,
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=20,
        action_joint_names={"toy": joints},
        action_low=torch.zeros(4),
        action_high=torch.ones(4),
        default_action=torch.zeros(4),
        instruction="Pick up the red cube.",
        access={"name": "visual", "observation": "rgb", "tools": "none"},
    )
    obs = SimpleNamespace(
        robots={
            "toy": SimpleNamespace(
                joint_pos=torch.zeros((1, 4)),
                root_state=torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            )
        },
        objects={},
        cameras={"front": SimpleNamespace(rgb=torch.zeros((1, 32, 32, 3), dtype=torch.uint8))},
    )
    return cfg, ctx, obs


def test_ee_delta_policy_turns_a_delta_into_joint_targets_and_holds_them(tmp_path, monkeypatch):
    cfg, ctx, obs = _ctx_and_obs(tmp_path, monkeypatch)
    policy = _ConstantVLA(repeat=3, tcp_offset=(0.0, 0.0, 0.0))
    policy.reset([0], [SimpleNamespace(prompt="Lift the block.")], ctx)
    acts = [policy.act(obs, ctx)[0]["toy"]["dof_pos_target"] for _ in range(4)]
    assert policy.calls == 2, "one model call per `repeat` simulator steps"
    assert policy.seen[0] == ((32, 32, 3), "Lift the block."), "the episode's own prompt, the camera frame"
    assert acts[0] == acts[1] == acts[2] and acts[3] == acts[0], (
        "held targets, then the same delta again from the same state"
    )
    assert acts[0]["finger_joint"] == 0.04, "gripper command 1 opens"
    arm = ArmModel(cfg, tcp_offset=(0.0, 0.0, 0.0))
    before = arm.fk(np.zeros(3))[0]
    after = arm.fk(np.array([acts[0][j] for j in arm.arm_joints]))[0]
    assert np.allclose(after - before, [0.0, 0.0, 0.02], atol=2e-3), "the tool moved +2 cm in z"


def test_ee_delta_policy_needs_a_camera_and_clips_large_steps(tmp_path, monkeypatch):
    cfg, ctx, obs = _ctx_and_obs(tmp_path, monkeypatch)

    class _Big(_ConstantVLA):
        def predict(self, image, instruction):
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    policy = _Big(max_step=0.05, tcp_offset=(0.0, 0.0, 0.0))
    policy.reset([0], [SimpleNamespace(prompt=None)], ctx)
    act = policy.act(obs, ctx)[0]["toy"]["dof_pos_target"]
    arm = ArmModel(cfg, tcp_offset=(0.0, 0.0, 0.0))
    moved = arm.fk(np.array([act[j] for j in arm.arm_joints]))[0] - arm.fk(np.zeros(3))[0]
    assert np.linalg.norm(moved) <= 0.05 + 2e-3, "a 1 m request is clipped to max_step"
    assert act["finger_joint"] == 0.0, "gripper command 0 closes"
    obs.cameras = {}
    policy._held.clear()
    with pytest.raises(ValueError, match="needs a camera image"):
        policy.act(obs, ctx)
