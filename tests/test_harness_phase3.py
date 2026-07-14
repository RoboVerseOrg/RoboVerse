"""Phase-3 tests: baseline policy adapters (scripted) produce spec-valid actions."""

from __future__ import annotations

import pytest
import torch

from roboverse_learn.eval.harness.adapters import HoldPosePolicy, RandomPolicy, ZeroActionPolicy
from roboverse_learn.eval.harness.embodiment import infer_embodiment
from roboverse_learn.eval.harness.obs import ObsBatch
from roboverse_learn.eval.harness.spec import derive_action_spec, derive_obs_spec


class _Act:
    def __init__(self, is_ee=False):
        self.is_ee = is_ee


class _Robot:
    def __init__(self, name, joints, ee_joints=()):
        self.name = name
        self.joint_limits = {j: (0.0, 1.0) for j in joints}
        self.actuators = {j: _Act(is_ee=(j in ee_joints)) for j in joints}
        self.gripper_open_q = None
        self.gripper_joint_name = None
        self.ee_body_name = f"{name}_hand"


def _franka():
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    grip = ["panda_finger_joint1", "panda_finger_joint2"]
    return _Robot("franka", arm + grip, ee_joints=grip)


def _obs(emb, obs_spec, b=3):
    tensors = {}
    for f in obs_spec.fields:
        if f.dtype == "uint8":
            tensors[f.key] = torch.zeros(b, *f.shape, dtype=torch.uint8)
        else:
            tensors[f.key] = torch.randn(b, *f.shape)
    return ObsBatch(obs_spec, torch.arange(b), tensors)


@pytest.mark.general
@pytest.mark.parametrize("cls", [ZeroActionPolicy, HoldPosePolicy, RandomPolicy])
def test_scripted_policy_produces_valid_action(cls):
    emb = infer_embodiment([_franka()])
    obs_spec = derive_obs_spec(emb, include_ee_pose=False)
    action_spec = derive_action_spec(emb, control="joint_pos")
    policy = cls()
    policy.bind(obs_spec, action_spec)
    policy.reset(torch.arange(3))
    obs = _obs(emb, obs_spec, b=3)
    action = policy.act(obs)
    action.validate()  # keys + shapes match the action spec
    assert set(action.tensors) == set(action_spec.keys())
    assert torch.equal(action.env_ids, obs.env_ids)


@pytest.mark.general
def test_hold_pose_echoes_joint_obs():
    emb = infer_embodiment([_franka()])
    obs_spec = derive_obs_spec(emb, include_ee_pose=False)
    action_spec = derive_action_spec(emb, control="joint_pos")
    p = HoldPosePolicy()
    p.bind(obs_spec, action_spec)
    obs = _obs(emb, obs_spec, b=2)
    action = p.act(obs)
    # hold-pose action equals the joint-space observation (canonical keys shared)
    assert torch.allclose(action.tensors["arm.joint_pos"], obs.tensors["arm.joint_pos"])


@pytest.mark.general
def test_zero_policy_is_zero():
    emb = infer_embodiment([_franka()])
    action_spec = derive_action_spec(emb, control="joint_pos")
    p = ZeroActionPolicy()
    p.bind(derive_obs_spec(emb, include_ee_pose=False), action_spec)
    action = p.act(_obs(emb, derive_obs_spec(emb, include_ee_pose=False), b=2))
    assert torch.count_nonzero(action.tensors["arm.joint_pos"]) == 0


@pytest.mark.general
@pytest.mark.parametrize("cls", [ZeroActionPolicy, HoldPosePolicy, RandomPolicy])
def test_card_matches_what_the_policy_emits(cls):
    # the PolicyCard is the negotiation contract: what a baseline advertises must be what the
    # harness derives for it and what it actually emits (asserting `chunk_len == 1` alone would
    # pass on a literal constant while the policy returned a chunk).
    emb = infer_embodiment([_franka()])
    obs_spec = derive_obs_spec(emb, include_ee_pose=False)
    action_spec = derive_action_spec(emb, control="joint_pos")
    card = cls().describe()
    # an empty advertised spec = "bind() will tell me"; it must negotiate against the derived one
    assert action_spec.compatible_with(card.produces_action).ok
    assert obs_spec.compatible_with(card.needs_obs).ok
    policy = cls()
    policy.bind(obs_spec, action_spec)
    action = policy.act(_obs(emb, obs_spec, b=2))
    # chunk_len=1 in the card => a FLAT (B, *shape) action, which is what the runner's 'none' path
    # requires; a chunked return here would silently degrade to no ensembling.
    assert card.produces_action.chunk_len == 1
    assert action.tensors["arm.joint_pos"].dim() == 2  # no horizon dim
    assert action.tensors["arm.joint_pos"].shape == (2, 7)
    action.validate()
