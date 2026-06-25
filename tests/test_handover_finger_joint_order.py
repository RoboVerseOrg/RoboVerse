"""Regression: ALOHA handover obs reads finger joints by sorted-name order.

``RobotState.joint_pos`` is ordered by sorted joint name. For the dual-arm ALOHA
that interleaves the two arms, so the four ``*_finger`` joints land at columns
2, 3, 10, 11 — NOT the last four columns (which are right-arm shoulder/waist/
wrist_angle/wrist_rotate).

The ``_observation`` method previously sliced ``robot_joint_pos[:, -4:]`` for the
finger-vs-box term, feeding arm joints into the policy observation. This pins the
corrected name-based resolution. Backend-free: a fake handler supplies the sorted
joint names and ``_observation`` is called on a hand-built state.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from metasim.types import ObjectState, RobotState

# Alphabetically sorted ALOHA joint names (matches roboverse_pack/robots/aloha.py).
_ALOHA_SORTED = [
    "left/elbow",
    "left/forearm_roll",
    "left/left_finger",
    "left/right_finger",
    "left/shoulder",
    "left/waist",
    "left/wrist_angle",
    "left/wrist_rotate",
    "right/elbow",
    "right/forearm_roll",
    "right/left_finger",
    "right/right_finger",
    "right/shoulder",
    "right/waist",
    "right/wrist_angle",
    "right/wrist_rotate",
]
_FINGER_COLS = [2, 3, 10, 11]  # where "*_finger" lands after sorting


def _identity_object():
    rs = torch.zeros(1, 13)
    rs[0, 3] = 1.0  # quaternion wxyz identity (matrix_from_quat needs a valid quat)
    return ObjectState(root_state=rs)


def _make_task_and_states(finger_val: float, arm_val: float):
    from roboverse_pack.tasks.mujoco_playground.handover import HandOver

    task = HandOver.__new__(HandOver)  # bypass __init__/backend
    task.robot_name = "aloha"
    task.num_envs = 1
    task.device = "cpu"
    # Deliberately do NOT set task._finger_joint_idx — it must come from the
    # class-level default so this also guards against it being (re)introduced as
    # a post-super().__init__() instance attribute (which RLTaskEnv.__init__'s
    # obs-probe would hit before it exists, crashing real construction).
    task.handler = SimpleNamespace(get_joint_names=lambda name, sort=True: _ALOHA_SORTED)

    jp = torch.full((1, 16), arm_val)
    for c in _FINGER_COLS:
        jp[0, c] = finger_val
    states = SimpleNamespace(
        objects={"box": _identity_object(), "target": _identity_object()},
        extras={
            "left_gripper_pos": torch.zeros(1, 3),
            "left_gripper_mat": torch.zeros(1, 9),
            "right_gripper_pos": torch.zeros(1, 3),
            "right_gripper_mat": torch.zeros(1, 9),
        },
        robots={"aloha": RobotState(root_state=torch.zeros(1, 13), joint_pos=jp, joint_vel=torch.zeros(1, 16))},
    )
    return task, states


def test_handover_obs_finger_term_uses_finger_columns():
    """The finger-vs-box obs term (obs cols 32:36) must come from the finger joints.

    Finger columns are set to 0.5 and every arm column to 9.0; the term is
    ``finger_joints - 0.04``. Correct -> ~0.46. The pre-fix ``[:, -4:]`` read arm
    columns 12-15 (=9.0) -> ~8.96, so this asserts the gap only the fix produces.
    """
    from roboverse_pack.tasks.mujoco_playground.handover import HandOver

    task, states = _make_task_and_states(finger_val=0.5, arm_val=9.0)
    obs = HandOver._observation(task, states)

    assert task._finger_joint_idx == _FINGER_COLS, (
        f"finger columns should resolve to {_FINGER_COLS} from sorted ALOHA joint names, got {task._finger_joint_idx}"
    )
    finger_box_diff = obs[0, 32:36]  # robot_joint_pos(16) + robot_joint_vel(16) -> finger term at 32:36
    assert torch.allclose(finger_box_diff, torch.full((4,), 0.46), atol=1e-4), (
        f"finger-vs-box obs term should be finger(0.5)-0.04=0.46, got {finger_box_diff.tolist()}; "
        f"a value near 8.96 means the obs is reading arm joints (the URDF last-4 bug)."
    )


def test_finger_idx_is_class_level_default():
    """`_finger_joint_idx` must be a class attribute so it exists before __init__ finishes.

    RLTaskEnv.__init__ probes `_observation` during construction; if the cache were
    only initialised after super().__init__(), the probe would AttributeError.
    """
    from roboverse_pack.tasks.mujoco_playground.handover import HandOver

    assert HandOver.__dict__.get("_finger_joint_idx", "MISSING") is None


def test_handover_does_not_slice_last_four_joints():
    """Source guard: the URDF-order ``[:, -4:]`` finger slice must not return."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "mujoco_playground" / "handover.py"
    ).read_text()
    assert "robot_joint_pos[:, -4:]" not in src, (
        "handover.py slices the last 4 joint columns as fingers; ALOHA fingers are at "
        "sorted columns 2,3,10,11 — resolve by name instead."
    )
