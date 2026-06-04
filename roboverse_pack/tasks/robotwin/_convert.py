"""Shared RoboTwin -> RoboVerse trajectory converter (single source of truth).

Both the replay example (``get_started/10_robotwin_aloha_replay.py``) and the
parity harness (``tools/robotwin_integration/parity_robotwin.py``) convert a
RoboTwin bridge pickle into RoboVerse's name-keyed ``*_v2`` format through this
module, so the joint mapping is defined in exactly one place.

RoboTwin demos are *single-embodiment bimanual*: one ALOHA-AgileX articulation
(two arx5 arms on an AgileX base) whose action spans both arms. RoboVerse
expresses this as the one-robot case of the name-keyed ``*_v2`` layout. The
14-D RoboTwin vector ``[L_arm(6), L_grip, R_arm(6), R_grip]`` maps onto the
embodiment's controlled joints; the gripper value is normalized ``[0, 1]``
(open -> closed) and denormalized to a joint target via the embodiment's
per-arm ``gripper_scale``.

The arm-joint mapping, gripper denormalization, and base pose here were audited
against RoboTwin's ``assets/embodiments/aloha-agilex/config.yml`` and the arx5
URDF and confirmed faithful (no sign flips, byte-identical gripper formula,
matching ``robot_pose``).
"""

from __future__ import annotations

from loguru import logger as log

from metasim.utils.demo_util.loader import save_traj_file

ROBOT_NAME = "aloha_agilex"
# ALOHA-AgileX home/base pose, from RoboTwin's embodiment config.yml (robot_pose).
ROBOT_POS = [0.0, -0.65, 0.0]
ROBOT_ROT = [0.707, 0.0, 0.0, 0.707]  # wxyz, 90 deg about z

_L_ARM = [f"fl_joint{i}" for i in range(1, 7)]
_R_ARM = [f"fr_joint{i}" for i in range(1, 7)]
# The 16 joints the embodiment actually controls: 12 arm + 4 gripper finger joints.
CONTROLLED_ARM = _L_ARM + _R_ARM
CONTROLLED_GRIPPER = ["fl_joint7", "fl_joint8", "fr_joint7", "fr_joint8"]
# The arx5 URDF exposes 38 active joints (wheels, mast, spare arm links); we drive
# only the 16 above and pin the rest to zero so set_states/set_dof_targets get a
# value for every joint.
_ALL_JOINTS = [
    "right_wheel", "left_wheel", "fl_castor_wheel", "fr_castor_wheel", "rr_castor_wheel", "rl_castor_wheel",
    "fl_joint1", "fr_joint1", "lr_joint1", "rr_joint1", "fl_wheel", "fr_wheel", "rr_wheel", "rl_wheel",
    "fl_joint2", "fr_joint2", "lr_joint2", "rr_joint2", "fl_joint3", "fr_joint3", "lr_joint3", "rr_joint3",
    "fl_joint4", "fr_joint4", "lr_joint4", "rr_joint4", "fl_joint5", "fr_joint5", "lr_joint5", "rr_joint5",
    "fl_joint6", "fr_joint6", "lr_joint6", "rr_joint6", "fl_joint7", "fl_joint8", "fr_joint7", "fr_joint8",
]  # fmt: skip


def vector_to_dof(vec, left_scale, right_scale, *, left_mimic=(1.0, 0.0), right_mimic=(1.0, 0.0)) -> dict:
    """Map RoboTwin's 14-D bimanual vector onto the embodiment's joint targets.

    ``vec`` is ``[L_arm(6), L_grip, R_arm(6), R_grip]``. Gripper values are the
    normalized ``[0, 1]`` (open -> closed) RoboTwin scalar; each is denormalized
    to the *driven* finger-joint target via ``gripper_scale = [open_val,
    closed_val]``. The opposing finger joint follows the embodiment's URDF
    ``mimic`` spec ``(multiplier, offset)``: ``joint8 = joint7 * mult + offset``.

    ``left_mimic`` / ``right_mimic`` default to ``(1.0, 0.0)`` — the ALOHA-AgileX
    unity mimic, where both finger joints take the same value (byte-identical to
    the previous behaviour). A future embodiment with a non-unity multiplier /
    non-zero offset passes its spec here instead of silently dropping it.
    """
    dof = {name: 0.0 for name in _ALL_JOINTS}
    for i, joint in enumerate(_L_ARM):
        dof[joint] = float(vec[i])
    for i, joint in enumerate(_R_ARM):
        dof[joint] = float(vec[7 + i])
    left_grip = left_scale[0] + float(vec[6]) * (left_scale[1] - left_scale[0])
    right_grip = right_scale[0] + float(vec[13]) * (right_scale[1] - right_scale[0])
    dof["fl_joint7"] = left_grip
    dof["fl_joint8"] = left_grip * float(left_mimic[0]) + float(left_mimic[1])
    dof["fr_joint7"] = right_grip
    dof["fr_joint8"] = right_grip * float(right_mimic[0]) + float(right_mimic[1])
    return dof


def bridge_to_v2(bridge: dict, out_path: str) -> None:
    """Convert a RoboTwin bridge pickle into a name-keyed ``*_v2`` dataset."""
    vectors = bridge["vectors"]
    ls, rs = bridge["left_gripper_scale"], bridge["right_gripper_scale"]
    init = {ROBOT_NAME: {"pos": ROBOT_POS, "rot": ROBOT_ROT, "dof_pos": vector_to_dof(vectors[0], ls, rs)}}
    # RoboTwin's block is a static box; include it for context (the manipulated
    # mesh object is omitted -- the bridge records only initial object poses).
    block = bridge["init_objects"].get("box")
    if block is not None:
        init["block"] = {"pos": block["pos"], "rot": block["rot"]}
    actions = [{"dof_pos_target": vector_to_dof(v, ls, rs)} for v in vectors[1:]]
    dataset = {
        ROBOT_NAME: [{"init_state": init, "actions": actions, "states": None}],
        "metadata": {
            "num_agents": 1,
            "agents": [ROBOT_NAME],
            "source": f"RoboTwin {bridge.get('task', '?')}",
            "source_seed": bridge.get("seed"),
        },
    }
    save_traj_file(dataset, out_path)
    log.info(f"Converted RoboTwin '{bridge.get('task')}' ({len(vectors)} frames) -> {out_path}")
