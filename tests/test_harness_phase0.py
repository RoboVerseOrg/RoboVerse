"""Phase-0 tests for the unified eval harness (roboverse_learn/eval/harness).

Pure / GPU-free: embodiment inference, typed specs + negotiation, carriers, and the
chunking/temporal-ensembling (incl. a regression vs the il temporal-agg algorithm).
"""

from __future__ import annotations

import pytest
import torch

from roboverse_learn.eval.harness import (
    ActionBatch,
    ActionChunk,
    ChainKind,
    ChunkScheduler,
    ObsBatch,
    Space,
    TemporalEnsembler,
    derive_action_spec,
    derive_obs_spec,
    infer_embodiment,
)
from roboverse_learn.eval.harness.embodiment import EmbodimentHints


class _Act:
    def __init__(self, is_ee=False):
        self.is_ee = is_ee


class _Robot:
    def __init__(self, name, joints, ee_joints=(), gripper_open_q=None):
        self.name = name
        self.joint_limits = {j: (0.0, 1.0) for j in joints}
        self.actuators = {j: _Act(is_ee=(j in ee_joints)) for j in joints}
        self.gripper_open_q = gripper_open_q
        self.gripper_joint_name = None
        self.ee_body_name = f"{name}_hand"


def _franka(name="franka"):
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    grip = ["panda_finger_joint1", "panda_finger_joint2"]
    return _Robot(name, arm + grip, ee_joints=grip)


# ------------------------------------------------------------------ embodiment
@pytest.mark.general
def test_embodiment_franka_single_arm():
    emb = infer_embodiment([_franka()])
    assert len(emb.arms) == 1 and len(emb.grippers) == 1
    assert emb.arms[0].dof == 7 and emb.grippers[0].dof == 2
    assert emb.arms[0].ee_body_name == "franka_hand"


@pytest.mark.general
def test_embodiment_is_ee_beats_name():
    # a gripper joint NOT named finger/gripper is still found via is_ee
    r = _Robot("r", ["j1", "j2", "j3", "grip_a", "grip_b"], ee_joints=["grip_a", "grip_b"])
    emb = infer_embodiment([r])
    assert emb.grippers[0].dof == 2 and set(emb.grippers[0].joint_names) == {"grip_a", "grip_b"}
    assert emb.arms[0].dof == 3


@pytest.mark.general
def test_embodiment_fallback_no_is_ee():
    # no is_ee markers -> name-token fallback finds the finger joints
    r = _Robot("r", ["shoulder", "elbow", "left_finger", "right_finger"])
    r.actuators = {}  # force fallback
    emb = infer_embodiment([r])
    assert emb.grippers and emb.grippers[0].dof == 2


@pytest.mark.general
def test_embodiment_humanoid_n_chains():
    joints = (
        [f"left_arm_{i}" for i in range(4)]
        + [f"right_arm_{i}" for i in range(4)]
        + [f"left_hip_{i}" for i in range(3)]
        + [f"right_hip_{i}" for i in range(3)]
        + ["torso", "head"]
    )
    emb = infer_embodiment([_Robot("h", joints)])
    kinds = {c.kind for c in emb.chains}
    assert {ChainKind.ARM, ChainKind.LEG, ChainKind.TORSO, ChainKind.HEAD} <= kinds
    assert len(emb.arms) == 2 and len(emb.by_kind(ChainKind.LEG)) == 2  # N-embodiment, no 2-arm ceiling


@pytest.mark.general
def test_embodiment_hints_override():
    r = _Robot("r", ["a", "b", "c", "d"])
    hints = EmbodimentHints(chains={"r": {"custom": (ChainKind.ARM, ("a", "b"))}})
    emb = infer_embodiment([r], hints=hints)
    assert emb.chain("custom").joint_names == ("a", "b")


@pytest.mark.general
def test_embodiment_multi_robot_disambiguation():
    emb = infer_embodiment([_franka("left"), _franka("right")])
    # two robots each contribute arm+gripper; names stay unique
    assert len(emb.chains) == len(set(c.name for c in emb.chains))
    assert len(emb.arms) == 2


# ------------------------------------------------------------------- spec
@pytest.mark.general
def test_derive_specs_canonical_keys():
    emb = infer_embodiment([_franka()])
    obs = derive_obs_spec(emb, cameras=[("head_cam", (128, 128))])
    assert "arm.joint_pos" in obs.keys()
    assert "arm.ee_pose" in obs.keys() and obs.field("arm.ee_pose").shape == (7,)
    assert "gripper.gripper" in obs.keys()
    assert obs.field("head_cam.rgb").shape == (128, 128, 3) and obs.field("head_cam.rgb").dtype == "uint8"
    act = derive_action_spec(emb, control="joint_pos")
    assert act.field("arm.joint_pos").shape == (7,)


@pytest.mark.general
def test_action_spec_ee_control():
    emb = infer_embodiment([_franka()])
    act = derive_action_spec(emb, control="ee_pose")
    assert act.field("arm.ee_pose").space == Space.EE_POSE and act.field("arm.ee_pose").shape == (7,)
    assert act.field("gripper.gripper").shape == (2,)


@pytest.mark.general
def test_obs_spec_omits_ee_pose_without_ee_body():
    # regression: derive_obs_spec emitted a REQUIRED <arm>.ee_pose for every arm even when the
    # RobotCfg has no ee_body_name (cartpole/h1/shadow_hand/allegro all do), and the adapter then
    # returned a constant identity pose forever. Don't emit a field the robot cannot provide.
    r = _franka()
    r.ee_body_name = None
    emb = infer_embodiment([r])
    assert "arm.ee_pose" not in derive_obs_spec(emb, include_ee_pose=True).keys()
    assert "arm.joint_pos" in derive_obs_spec(emb, include_ee_pose=True).keys()
    # ...and an ee_pose *action* on such an arm is a hard error, not a silent identity target
    with pytest.raises(ValueError, match="ee_body_name"):
        derive_action_spec(emb, control="ee_pose")
    # with an ee body, the field is emitted and carries its (world) frame
    emb2 = infer_embodiment([_franka()])
    f = derive_obs_spec(emb2, include_ee_pose=True).field("arm.ee_pose")
    assert f.shape == (7,) and f.frame == "world"


@pytest.mark.general
@pytest.mark.parametrize("control", ["joint_vel", "effort", "bogus"])
def test_derive_action_spec_rejects_unsupported_control(control):
    # regression: control="joint_vel"/"effort" were accepted but produced an ActionSpec whose
    # fields were all Space.JOINT_POS keyed "<chain>.joint_pos" — an incoherent contract.
    emb = infer_embodiment([_franka()])
    with pytest.raises(ValueError, match="unsupported control"):
        derive_action_spec(emb, control=control)


@pytest.mark.general
def test_action_spec_negotiation_control_and_fields():
    from roboverse_learn.eval.harness.spec import ActionSpec, FieldSpec

    emb = infer_embodiment([_franka()])
    env_spec = derive_action_spec(emb, control="joint_pos")
    # a policy that advertises nothing (binds later) is trivially compatible
    assert env_spec.compatible_with(ActionSpec(())).ok
    # a policy that declares control="ee_pose" must NOT silently get a joint_pos spec
    ee = derive_action_spec(emb, control="ee_pose")
    m = env_spec.compatible_with(ee)
    assert not m.ok and any("ee_pose" in e for e in m.errors)
    # a policy that only commands the arm, forgetting the gripper, is rejected
    partial = ActionSpec((FieldSpec("arm.joint_pos", Space.JOINT_POS, (7,), chain="arm"),), control="joint_pos")
    m2 = env_spec.compatible_with(partial)
    assert not m2.ok and any("gripper.gripper" in e for e in m2.errors)
    # a matching declaration passes
    assert env_spec.compatible_with(derive_action_spec(emb, control="joint_pos")).ok


@pytest.mark.general
def test_spec_negotiation_match_and_mismatch():
    emb = infer_embodiment([_franka()])
    full = derive_obs_spec(emb, cameras=[("cam", (84, 84))])
    # policy needs only arm.joint_pos + cam -> compatible
    needs = full.subset(["arm.joint_pos", "cam.rgb"])
    m = full.compatible_with(needs)
    assert m.ok and {op.key for op in m.plan} == {"arm.joint_pos", "cam.rgb"}
    # policy needs a field the env doesn't produce -> incompatible with actionable error
    from roboverse_learn.eval.harness.spec import FieldSpec, ObsSpec

    bad = ObsSpec((FieldSpec("wrist_cam.rgb", Space.RGB, (84, 84, 3)),))
    m2 = full.compatible_with(bad)
    assert not m2.ok and any("wrist_cam.rgb" in e for e in m2.errors)


@pytest.mark.general
def test_spec_optional_field_dropped():
    from roboverse_learn.eval.harness.spec import FieldSpec, ObsSpec

    prod = ObsSpec((FieldSpec("arm.joint_pos", Space.JOINT_POS, (7,)),))
    cons = ObsSpec((
        FieldSpec("arm.joint_pos", Space.JOINT_POS, (7,)),
        FieldSpec("task.language", Space.TASK, (), dtype="str", required=False),
    ))
    m = prod.compatible_with(cons)
    assert m.ok and any(op.op == "drop_optional" for op in m.plan)


# ------------------------------------------------------------------- carriers
@pytest.mark.general
def test_obsbatch_validate_shapes():
    emb = infer_embodiment([_franka()])
    spec = derive_obs_spec(emb, include_ee_pose=False)
    b = 4
    tensors = {"arm.joint_pos": torch.zeros(b, 7), "gripper.gripper": torch.zeros(b, 2)}
    ob = ObsBatch(spec, torch.arange(b), tensors).validate()
    assert ob.batch_size == 4
    assert ob.index(torch.tensor([0, 2])).batch_size == 2
    with pytest.raises(ValueError):
        ObsBatch(
            spec, torch.arange(b), {"arm.joint_pos": torch.zeros(b, 6), "gripper.gripper": torch.zeros(b, 2)}
        ).validate()


@pytest.mark.general
def test_actionbatch_chunk_detection():
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=8)
    b = 3
    chunked = {"arm.joint_pos": torch.zeros(b, 8, 7), "gripper.gripper": torch.zeros(b, 8, 2)}
    assert ActionBatch(spec, torch.arange(b), chunked).is_chunked
    ch = ActionChunk.from_batch(ActionBatch(spec, torch.arange(b), chunked))
    assert ch.horizon == 8


# ------------------------------------------------------------------- chunking
def _il_temporal_agg_ref(pushed, k, E, H, step_query):
    """Faithful reimplementation of il BaseEvalRunner.get_temporal_agg_action (single env)."""
    dim = pushed[0][1].shape[1]
    buf = torch.zeros(E, E + H, dim)
    for t, ch in pushed:
        buf[t, t : t + H] = ch
    col = buf[:, step_query]  # (E, dim)
    populated = torch.all(col != 0, dim=1)
    sel = col[populated]
    ti = torch.arange(sel.shape[0]).float()
    w = torch.exp(k * ti)
    w = w / w.sum()
    return (sel * w.unsqueeze(-1)).sum(0)


@pytest.mark.general
def test_temporal_ensembler_matches_il():
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    E, H, k = 20, 4, 0.05
    ens = TemporalEnsembler(action_spec=spec, num_envs=1, k=k)
    torch.manual_seed(0)
    # push overlapping chunks at steps 0..5 (nonzero so il's !=0 populated test is unambiguous)
    pushed = []
    for t in range(6):
        arm = torch.randn(1, H, 7) + 3.0
        grip = torch.randn(1, H, 2) + 3.0
        ens.push(t, ActionChunk({"arm.joint_pos": arm, "gripper.gripper": grip}, H), torch.tensor([0]))
        pushed.append((t, torch.cat([arm[0], grip[0]], dim=1)))  # (H, 9) combined for the ref
    step_q = 5
    out = ens.action_for(step_q, torch.tensor([0]))
    got = torch.cat([out["arm.joint_pos"][0], out["gripper.gripper"][0]])
    ref = _il_temporal_agg_ref(pushed, k, E, H, step_q)
    assert torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max()}"


@pytest.mark.general
def test_chunk_scheduler_pops_in_order():
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=3)
    sch = ChunkScheduler(action_spec=spec, num_envs=1)
    assert sch.needs_query().tolist() == [0]  # starts exhausted
    arm = torch.stack([torch.full((7,), float(i)) for i in range(3)]).unsqueeze(0)  # (1,3,7)
    grip = torch.zeros(1, 3, 2)
    sch.push(ActionChunk({"arm.joint_pos": arm, "gripper.gripper": grip}, 3), torch.tensor([0]))
    vals = [sch.action_for(torch.tensor([0]))["arm.joint_pos"][0, 0].item() for _ in range(3)]
    assert vals == [0.0, 1.0, 2.0]  # emits chunk steps in order
    assert sch.needs_query().tolist() == [0]  # exhausted again after H pops


@pytest.mark.general
def test_chunk_scheduler_short_chunk_requeries():
    # regression: a chunk shorter than H must re-query at its real tail, not emit stale data
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    sch = ChunkScheduler(action_spec=spec, num_envs=1)
    long = torch.stack([torch.full((7,), 100.0 + i) for i in range(4)]).unsqueeze(0)
    sch.push(ActionChunk({"arm.joint_pos": long, "gripper.gripper": torch.zeros(1, 4, 2)}, 4), torch.tensor([0]))
    sch.action_for(torch.tensor([0]))  # consume 1 of the long chunk
    short = torch.stack([torch.full((7,), float(i)) for i in range(2)]).unsqueeze(0)  # horizon 2
    sch.push(ActionChunk({"arm.joint_pos": short, "gripper.gripper": torch.zeros(1, 2, 2)}, 2), torch.tensor([0]))
    vals = [sch.action_for(torch.tensor([0]))["arm.joint_pos"][0, 0].item() for _ in range(2)]
    assert vals == [0.0, 1.0]  # only the 2 valid actions
    assert sch.needs_query().tolist() == [0]  # re-query after 2, NOT emit stale 102/103


@pytest.mark.general
def test_temporal_ensembler_reset_no_leak():
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=2)
    ens = TemporalEnsembler(action_spec=spec, num_envs=1, k=0.05)
    ens.push(
        0,
        ActionChunk({"arm.joint_pos": torch.full((1, 2, 7), 5.0), "gripper.gripper": torch.zeros(1, 2, 2)}, 2),
        torch.tensor([0]),
    )
    ens.reset(torch.tensor([0]))  # new episode; stale buffer must not leak
    ens.push(
        3,
        ActionChunk({"arm.joint_pos": torch.full((1, 2, 7), 9.0), "gripper.gripper": torch.zeros(1, 2, 2)}, 2),
        torch.tensor([0]),
    )
    out = ens.action_for(3, torch.tensor([0]))["arm.joint_pos"][0]
    assert torch.allclose(out, torch.full((7,), 9.0)), out  # only the post-reset prediction


@pytest.mark.general
def test_temporal_ensembler_chunk_len_1():
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=1)
    ens = TemporalEnsembler(action_spec=spec, num_envs=1, k=0.05)
    a = torch.randn(1, 1, 7) + 3.0
    ens.push(2, ActionChunk({"arm.joint_pos": a, "gripper.gripper": torch.randn(1, 1, 2) + 3.0}, 1), torch.tensor([0]))
    out = ens.action_for(2, torch.tensor([0]))["arm.joint_pos"][0]
    assert torch.allclose(out, a[0, 0], atol=1e-6)  # single prediction -> itself


@pytest.mark.general
def test_temporal_ensembler_buffer_is_bounded_by_horizon():
    # regression: the buffer used to be (num_envs, E, E+H, dim) — O(N*E^2*D), multi-GB at
    # num_envs=64 / max_episode_steps=1000. It must be a ring over the chunk horizon instead,
    # i.e. independent of episode length.
    emb = infer_embodiment([_franka()])
    spec = derive_action_spec(emb, control="joint_pos", chunk_len=4)
    ens = TemporalEnsembler(action_spec=spec, num_envs=8, k=0.05)
    assert ens._buf["arm.joint_pos"].shape == (8, 4, 4, 7)  # (N, H, H, dim), no episode_len term
    total = sum(b.numel() for b in ens._buf.values())
    assert total == 8 * 4 * 4 * (7 + 2)
    # and it still ensembles correctly far into a long episode (ring wraps, no episode_len bound)
    for t in (996, 997, 998):
        chunk = ActionChunk(
            {"arm.joint_pos": torch.full((8, 4, 7), float(t)), "gripper.gripper": torch.zeros(8, 4, 2)}, 4
        )
        ens.push(t, chunk, torch.arange(8))
    out = ens.action_for(998, torch.arange(8))["arm.joint_pos"]
    assert out.shape == (8, 7)
    assert out.min() > 996.0 and out.max() < 998.0  # blended across the 3 live predictions


# ------------------------------------------------------- embodiment regressions (fixed bugs)
class _NsRobot(_Robot):
    """Robot whose joints carry a namespace prefix (aloha-style)."""


@pytest.mark.general
def test_embodiment_namespace_side_and_jaws():
    # aloha-style: side is the '/' namespace; left_finger/right_finger are one gripper's jaws
    joints = []
    ee = []
    for side in ("left", "right"):
        joints += [f"{side}/waist", f"{side}/shoulder", f"{side}/elbow", f"{side}/wrist"]
        joints += [f"{side}/left_finger", f"{side}/right_finger"]
        ee += [f"{side}/left_finger", f"{side}/right_finger"]
    emb = infer_embodiment([_Robot("aloha", joints, ee_joints=ee)])
    assert len(emb.arms) == 2 and len(emb.grippers) == 2
    for g in emb.grippers:
        assert g.dof == 2  # both jaws in ONE gripper, not split by finger name


@pytest.mark.general
def test_embodiment_declared_gripper_outside_joint_limits_raises():
    # BLOCKER regression: the gripper joints live outside joint_limits, so the gripper used to be
    # silently DROPPED — the action spec became ('arm.joint_pos',) only and every pick task was
    # unsolvable by construction while reporting success_rate 0.0 with no error. Must raise.
    r = _Robot("ur", ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"])
    r.actuators = {}
    r.gripper_open_q = [0.0] * 6  # the cfg declares a gripper it cannot expose
    with pytest.raises(ValueError, match="declares a gripper"):
        infer_embodiment([r])
    # EmbodimentHints is the documented escape hatch: naming the chain explicitly must not raise
    # (here the gripper is genuinely absent, so an operator declaring the arm-only truth is fine).
    hints = EmbodimentHints(chains={"ur": {"gripper": (ChainKind.GRIPPER, ("shoulder_pan",))}})
    assert len(infer_embodiment([r], hints=hints).grippers) == 1


@pytest.mark.general
def test_embodiment_no_gripper_open_q_count_guess():
    # ...and a robot that declares NO gripper keeps all its joints in the arm (no last-N guess).
    r = _Robot("ur", ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"])
    r.actuators = {}
    emb = infer_embodiment([r])
    assert len(emb.grippers) == 0 and emb.arms[0].dof == 6


@pytest.mark.general
@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [
        (("LF", "LH", "RF", "RH"), {"front_left_leg", "rear_left_leg", "front_right_leg", "rear_right_leg"}),
        (("FL", "FR", "RL", "RR"), {"front_left_leg", "front_right_leg", "rear_left_leg", "rear_right_leg"}),
    ],
)
def test_embodiment_quadruped_four_legs(prefixes, expected):
    # regression: FL/FR/RL/RR (go2-style) matched no side token, so all 12 joints collapsed into
    # ONE leg chain. Both quadruped naming conventions must split into four legs.
    joints = [f"{s}_{j}" for s in prefixes for j in ("HAA", "HFE", "KFE")]
    r = _Robot("quad", joints)
    r.actuators = {}
    r.ee_body_name = None
    emb = infer_embodiment([r])
    assert {c.kind for c in emb.chains} == {ChainKind.LEG}
    assert {c.name for c in emb.chains} == expected
    assert all(c.dof == 3 for c in emb.chains) and sum(c.dof for c in emb.chains) == 12


@pytest.mark.general
def test_embodiment_unrecognized_joints_are_other_not_arm():
    # a robot with no manipulator evidence (no ee_body_name / is_ee / gripper name / arm token)
    # must NOT have its unrecognized joints relabelled "arm"
    r = _Robot("cart", ["slider_to_cart", "cart_to_pole"])
    r.actuators = {}
    r.ee_body_name = None
    emb = infer_embodiment([r])
    assert [(c.name, c.kind, c.dof) for c in emb.chains] == [("other", ChainKind.OTHER, 2)]
    assert not emb.arms
    # ...but the OTHER chain stays controllable in joint space
    assert derive_action_spec(emb).keys() == ("other.joint_pos",)


# --------------------------------------------- embodiment on the real in-repo robot cfgs
@pytest.mark.general
@pytest.mark.parametrize(
    ("robot_name", "expected"),
    [
        ("franka", {"arm": 7, "gripper": 2}),
        ("cartpole", {"other": 2}),  # was: arm(2)
        ("shadow_hand", {"other": 24}),  # was: arm(24)
        ("allegro_hand", {"other": 16}),  # was: arm(12) + gripper(4) (the "thumb" joints)
        ("go2", {"front_left_leg": 3, "front_right_leg": 3, "rear_left_leg": 3, "rear_right_leg": 3}),  # was: leg(12)
        ("h1", {"left_arm": 4, "right_arm": 4, "left_leg": 5, "right_leg": 5, "torso": 1}),
    ],
)
def test_embodiment_real_robot_cfgs(robot_name, expected):
    from metasim.utils.setup_util import get_robot

    emb = infer_embodiment([get_robot(robot_name)])
    assert {c.name: c.dof for c in emb.chains} == expected


@pytest.mark.general
@pytest.mark.parametrize("robot_name", ["ur5e_2f85", "kinova_gen3"])
def test_embodiment_real_cfg_with_unexposed_gripper_raises(robot_name):
    # both set gripper_open_q but keep no gripper joint in joint_limits
    from metasim.utils.setup_util import get_robot

    with pytest.raises(ValueError, match=robot_name):
        infer_embodiment([get_robot(robot_name)])


# ------------------------------------------------------- spec negotiation (dtype/frame)
@pytest.mark.general
def test_spec_dtype_mismatch_records_cast():
    from roboverse_learn.eval.harness.spec import FieldSpec, ObsSpec

    prod = ObsSpec((FieldSpec("cam.rgb", Space.RGB, (84, 84, 3), dtype="uint8"),))
    cons = ObsSpec((FieldSpec("cam.rgb", Space.RGB, (84, 84, 3), dtype="float32"),))
    m = prod.compatible_with(cons)
    assert m.ok and any(op.op == "cast" and "uint8->float32" in op.note for op in m.plan)


@pytest.mark.general
def test_spec_frame_mismatch_errors():
    from roboverse_learn.eval.harness.spec import FieldSpec, ObsSpec

    prod = ObsSpec((FieldSpec("arm.ee_pose", Space.EE_POSE, (7,), frame="world"),))
    cons = ObsSpec((FieldSpec("arm.ee_pose", Space.EE_POSE, (7,), frame="robot_base"),))
    assert not prod.compatible_with(cons).ok
    # a None (unspecified) frame is a wildcard -> compatible
    cons2 = ObsSpec((FieldSpec("arm.ee_pose", Space.EE_POSE, (7,)),))
    assert prod.compatible_with(cons2).ok
