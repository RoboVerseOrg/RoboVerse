"""Regression: ManiSkill dense rewards index Franka joints by the sorted-name contract.

MetaSim reports ``RobotState.joint_pos`` / ``joint_vel`` in alphabetically-sorted
joint-name order. For the Franka Panda that order is
``[panda_finger_joint1, panda_finger_joint2, panda_joint1..panda_joint7]`` — i.e.
the two gripper fingers are columns 0-1 and the seven arm joints are columns 2-8
(see ``roboverse_pack/robots/franka_cfg.py`` ``ee_joint_indices = [0, 1]``).

The dense reward functions used to slice ``joint_pos[:, 7:9]`` for the gripper and
``joint_vel[:, :7]`` for the arm — the *URDF* declaration order — so ``is_grasped``,
``gripper_width`` and the static/velocity terms were computed from the wrong joints
on every backend. These tests pin the corrected indexing without needing a
simulator backend or downloaded assets (``_reward`` is a pure function of the
passed ``env_states``).

Backend-free by design: no existing ``tests/test_maniskill_*`` file is usable here
because they all ``pytest.importorskip("mani_skill")`` at module load.
"""

from __future__ import annotations

import torch

from metasim.types import ObjectState, RobotState


def _pick_cube_states(finger_val: float, arm_tail_val: float = 1.5):
    """Build a minimal env_states for PickCube where the cube sits at the TCP.

    Fingers occupy sorted columns 0-1; ``arm_tail_val`` is written to columns 7-8
    (the columns the buggy code mistook for the gripper) so a regression cannot
    pass by reading the arm. ``finger_val`` per finger: 0.01 -> closed/grasped,
    0.04 -> open.
    """
    import roboverse_pack.tasks.maniskill.pick_cube_v1 as M

    jp = torch.zeros(1, 9)
    jp[0, 0] = finger_val
    jp[0, 1] = finger_val
    jp[0, 7] = arm_tail_val
    jp[0, 8] = arm_tail_val
    body_state = torch.zeros(1, 2, 13)
    body_state[0, 1, 3] = 1.0  # panda_hand quaternion = wxyz identity, at origin
    rs = RobotState(
        root_state=torch.zeros(1, 13),
        body_names=["panda_link0", "panda_hand"],
        body_state=body_state,
        joint_pos=jp,
        joint_vel=torch.zeros(1, 9),
    )
    cube = torch.zeros(1, 13)
    cube[0, :3] = torch.tensor(M._TCP_OFFSET)  # cube exactly at the TCP -> reaching satisfied
    goal = torch.zeros(1, 13)
    goal[0, 2] = 0.5  # goal far away -> not placed -> no success override

    class _S:
        pass

    s = _S()
    s.robots = {"franka": rs}
    s.objects = {"cube": ObjectState(root_state=cube), "goal_site": ObjectState(root_state=goal)}
    return s


def test_pick_cube_grasp_reward_reads_finger_columns():
    """Closing the fingers (sorted cols 0-1) must register as a grasp.

    On the pre-fix code (``joint_pos[:, 7:9]``) the reward read arm columns 7-8
    (here a large constant), so it never saw the grasp and both branches scored
    identically — this asserts the gap that only the corrected indexing produces.
    """
    from roboverse_pack.tasks.maniskill.pick_cube_v1 import PickCubeV1DenseTask

    grasped = float(PickCubeV1DenseTask._reward(None, _pick_cube_states(0.01))[0])
    ungrasped = float(PickCubeV1DenseTask._reward(None, _pick_cube_states(0.04))[0])

    assert grasped > ungrasped + 0.9, (
        f"grasp (fingers closed) should raise the reward via the is_grasped term; "
        f"got grasped={grasped:.4f} ungrasped={ungrasped:.4f}. A near-zero gap means "
        f"the reward is reading arm columns, not the gripper fingers (cols 0-1)."
    )


def test_no_maniskill_reward_uses_urdf_order_finger_slice():
    """Guard the whole dense-reward family against the URDF-order regression.

    The five sibling tasks (stack_cube, poke_cube, place_sphere, pick_single_ycb,
    lift_peg_upright) share the same Franka layout but build different object
    states, so this asserts the corrected slice convention at the source level
    rather than re-deriving each task's full env_states.
    """
    import pathlib

    pack = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "maniskill"
    offenders = []
    for py in pack.glob("*.py"):
        text = py.read_text()
        if "rs.joint_pos[:, 7:9]" in text or "rs.joint_vel[:, :7]" in text:
            offenders.append(py.name)
    assert not offenders, (
        f"these files slice Franka joints in URDF order (gripper 7:9 / arm :7) "
        f"instead of the sorted-name order (gripper 0:2 / arm 2:9): {offenders}"
    )
