"""Programmatically assemble standing-pose scene MJCFs for mjlab humanoids.

mjlab ships the raw robot MJCFs without actuators or a ground plane —
both are added at task-cfg time by their manager-based env config. For
our parity / visualization story we need a *complete* scene we can
load directly into MuJoCo (and into MetaSim/MuJoCo).

This builder uses `mujoco.MjSpec` to load the bare robot, add a floor,
add per-joint position actuators with mjlab-derived (kp, kv), and
emit a keyframe matching mjlab's standing pose. We then serialise the
compiled spec back to XML (with absolute mesh paths inlined) so any
downstream tool — MetaSim, raw mujoco, an external viewer — can load
the result without worrying about include paths.

Run as a script to (re)build the cached scenes:
    python -m tools.mjlab_integration.scenes.build_stand_scenes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

_MJLAB = os.environ.get("MJLAB_DIR", os.path.expanduser("~/projects/mjlab"))
MJLAB_ROOT = Path(_MJLAB)
CACHE_DIR = Path(__file__).parent / "_cache"


# (joint_name, kp, kv, effort_limit, armature) — replicates mjlab's
# BuiltinPositionActuatorCfg + ElectricActuator + reflected_inertia setup
# from unitree_g1/g1_constants.py. kp = armature*omega^2,
# kv = 2*zeta*armature*omega with omega=10*2pi rad/s, zeta=2.0.
#
# IMPORTANT: armature is set on the joint (via dof_armature) not the
# actuator. The PD law above produces ~10 Hz natural frequency when
# armature equals the dof's effective inertia.
#
# Motor classes:
#   5020   : arms (shoulder/elbow) and some lighter joints
#   7520_14: hip pitch/yaw, waist yaw
#   7520_22: knees, hip roll, ankles, waist roll/pitch, shoulder pitch
#   4010   : wrists
# Stiffer-than-mjlab kp/kv for the static-hold demo. mjlab's 10 Hz
# natural-frequency tuning is matched to its learned policy; for our
# scripted-hold rollout we need PD that can absorb gravity without
# letting the robot collapse. We multiply leg + waist + shoulder
# pitch kp by ~20× (and kv by ~4×) so the bent-knee keyframe holds
# under static load. The actuator effort_limit caps the torque so we
# can't violate physics — the limits stay at mjlab's values.
_G1_5020 = dict(kp=80, kv=4, eff=25, arm=0.00434)
_G1_7520_14 = dict(kp=1500, kv=30, eff=88, arm=0.01257)
_G1_7520_22 = dict(kp=3000, kv=50, eff=139, arm=0.03212)
_G1_4010 = dict(kp=40, kv=2, eff=5, arm=0.00425)

G1_ACTUATORS: list[tuple[str, float, float, float, float]] = [
    # legs — hip pitch/yaw use 7520_14, hip roll + knee + ankle use 7520_22
    ("left_hip_pitch_joint", _G1_7520_14["kp"], _G1_7520_14["kv"], _G1_7520_14["eff"], _G1_7520_14["arm"]),
    ("left_hip_roll_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("left_hip_yaw_joint", _G1_7520_14["kp"], _G1_7520_14["kv"], _G1_7520_14["eff"], _G1_7520_14["arm"]),
    ("left_knee_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("left_ankle_pitch_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("left_ankle_roll_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("right_hip_pitch_joint", _G1_7520_14["kp"], _G1_7520_14["kv"], _G1_7520_14["eff"], _G1_7520_14["arm"]),
    ("right_hip_roll_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("right_hip_yaw_joint", _G1_7520_14["kp"], _G1_7520_14["kv"], _G1_7520_14["eff"], _G1_7520_14["arm"]),
    ("right_knee_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("right_ankle_pitch_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("right_ankle_roll_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    # waist
    ("waist_yaw_joint", _G1_7520_14["kp"], _G1_7520_14["kv"], _G1_7520_14["eff"], _G1_7520_14["arm"]),
    ("waist_roll_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("waist_pitch_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    # arms — shoulder pitch uses 7520_22, others 5020; wrists 4010
    ("left_shoulder_pitch_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("left_shoulder_roll_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("left_shoulder_yaw_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("left_elbow_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("left_wrist_roll_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
    ("left_wrist_pitch_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
    ("left_wrist_yaw_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
    ("right_shoulder_pitch_joint", _G1_7520_22["kp"], _G1_7520_22["kv"], _G1_7520_22["eff"], _G1_7520_22["arm"]),
    ("right_shoulder_roll_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("right_shoulder_yaw_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("right_elbow_joint", _G1_5020["kp"], _G1_5020["kv"], _G1_5020["eff"], _G1_5020["arm"]),
    ("right_wrist_roll_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
    ("right_wrist_pitch_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
    ("right_wrist_yaw_joint", _G1_4010["kp"], _G1_4010["kv"], _G1_4010["eff"], _G1_4010["arm"]),
]

# Straight-leg "I-pose" — statically stable because COM sits over the
# foot support polygon. mjlab's KNEES_BENT_KEYFRAME is for the trained
# locomotion policy; without a learned balance controller, a bent-knee
# humanoid tips over within ~2 s even under perfect PD. With straight
# legs + slightly outward arms the bot just stands there.
G1_STANDING_POSE: dict[str, float] = {
    # legs: zero everywhere — full-length stance.
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
# Trunk height for the straight-leg pose. The g1 pelvis-to-foot-sole
# distance is ~0.79 m with zero leg joints; we add 1 cm clearance.
G1_TRUNK_Z = 0.80


# go1's actuator parameters mirror mjlab/asset_zoo/robots/unitree_go1/
# go1_constants.py:55 (10Hz natural freq, damping ratio 2). All 12 joints
# share the same Unitree A1 motor → effectively the same kp/kv/armature.
# Same rationale as G1: stiffer-than-mjlab for the scripted static-hold
# demo. mjlab's learned policy is what makes the 10 Hz tuning work; for
# raw scripted hold we need PD that absorbs gravity.
_GO1 = dict(kp=300, kv=15, eff=23.7, arm=0.00402)
GO1_ACTUATORS: list[tuple[str, float, float, float, float]] = []
for _prefix in ("FR", "FL", "RR", "RL"):
    GO1_ACTUATORS.extend([
        (f"{_prefix}_hip_joint", _GO1["kp"], _GO1["kv"], _GO1["eff"], _GO1["arm"]),
        (f"{_prefix}_thigh_joint", _GO1["kp"], _GO1["kv"], _GO1["eff"], _GO1["arm"]),
        (f"{_prefix}_calf_joint", _GO1["kp"], _GO1["kv"], 35.55, _GO1["arm"]),
    ])

# Statically-stable quadruped stance — straight thighs, slightly bent
# knees so each foot sits flat on the floor. mjlab's
# "thigh=0.9, calf=-1.8" is again a learned-policy crouch that isn't
# statically stable for a scripted hold.
GO1_STANDING_POSE: dict[str, float] = {}
for _prefix in ("FR", "FL", "RR", "RL"):
    GO1_STANDING_POSE[f"{_prefix}_hip_joint"] = 0.0
    GO1_STANDING_POSE[f"{_prefix}_thigh_joint"] = 0.8
    GO1_STANDING_POSE[f"{_prefix}_calf_joint"] = -1.6
GO1_TRUNK_Z = 0.32


YAM_ACTUATORS: list[tuple[str, float, float, float, float]] = [
    ("joint1", 200, 8, 35, 0.02),
    ("joint2", 200, 8, 35, 0.02),
    ("joint3", 120, 5, 18, 0.01),
    ("joint4", 100, 4, 18, 0.005),
    ("joint5", 60, 3, 10, 0.002),
    ("joint6", 40, 2, 10, 0.002),
]
YAM_STANDING_POSE: dict[str, float] = {
    "joint1": 0.0,
    "joint2": -0.8,
    "joint3": 1.5,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
}


def _add_floor(spec: mujoco.MjSpec, z: float) -> None:
    """Append a flat ground geom + a light."""
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.05],
        pos=[0, 0, z],
        rgba=[0.7, 0.7, 0.8, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )


def _add_position_actuators(spec: mujoco.MjSpec, actuators: list[tuple[str, float, float, float, float]]) -> None:
    """For each (joint_name, kp, kv, effort_limit, armature), wire a PD actuator.

    mjlab's `BuiltinPositionActuatorCfg` maps to:
      * MuJoCo `general` actuator with affine bias on (qpos - target),
        equivalent to a `<position kp=kp kv=kv>` shorthand.
      * The joint's `armature` is set so kp/kv land at the natural
        frequency they were designed for (10 Hz).

    We mirror both pieces here.
    """
    # First pass: armature on the joints.
    joints_by_name = {j.name: j for j in spec.joints}
    for name, _kp, _kv, _effort, arm in actuators:
        j = joints_by_name.get(name)
        if j is not None:
            j.armature = arm

    # Second pass: actuators.
    for name, kp, kv, effort, _arm in actuators:
        a = spec.add_actuator(
            name=f"{name}_act",
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-3.14159, 3.14159],
            forcerange=[-effort, effort],
            forcelimited=True,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        a.gainprm[0] = kp
        a.biasprm[1] = -kp
        a.biasprm[2] = -kv


def _add_keyframe(
    spec: mujoco.MjSpec,
    name: str,
    trunk_z: float | None,
    pose: dict[str, float],
    actuator_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos, ctrl) that match mjlab's standing keyframe."""
    # Compile first to discover qpos layout (free joint takes 7 slots,
    # hinge/slide take 1 each).
    model = spec.compile()
    qpos = np.zeros(model.nq, dtype=np.float64)
    if trunk_z is not None:
        # Find the freejoint (jnt_type == JNT_FREE) and place the trunk.
        for jid in range(model.njnt):
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                addr = int(model.jnt_qposadr[jid])
                qpos[addr + 0] = 0.0
                qpos[addr + 1] = 0.0
                qpos[addr + 2] = trunk_z
                qpos[addr + 3] = 1.0
                qpos[addr + 4] = 0.0
                qpos[addr + 5] = 0.0
                qpos[addr + 6] = 0.0
                break
    for jname, val in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        addr = int(model.jnt_qposadr[jid])
        qpos[addr] = val
    ctrl = np.array([pose[j] for j in actuator_order], dtype=np.float64)

    # Add a keyframe via spec so callers can `mj_resetDataKeyframe`.
    key = spec.add_key(name=name)
    key.qpos = qpos.tolist()
    key.ctrl = ctrl.tolist()
    return qpos, ctrl


def _weld_root_to_world(spec: mujoco.MjSpec, trunk_z: float | None) -> None:
    """Replace the robot's free root joint with a weld to the world.

    Used by the `_pendant` scenes — without an RL policy, a free-floating
    humanoid will tip over within a couple seconds even with stiff PD. The
    pendant variant proves the PD + actuator + scene plumbing works by
    holding the robot in mid-air and exercising the joints; the
    `_freefall` variant complements it by showing what happens when
    gravity + contact run unsupervised.

    Also lifts the root body to `trunk_z` so the robot hangs at standing
    height instead of clipping into the ground.
    """
    free_parent_body = None
    for jnt in list(spec.joints):
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
            free_parent_body = jnt.parent
            spec.delete(jnt)
            break
    if free_parent_body is not None and trunk_z is not None:
        free_parent_body.pos = [0.0, 0.0, trunk_z]


def build_compiled_model(robot: str, *, pin_root: bool = False) -> mujoco.MjModel:
    """Build the scene in-memory and return the compiled MjModel.

    Skips writing XML to disk so we don't have to chase down meshdir
    relative-path issues — the spec resolves them at compile time.
    """
    spec = _build_spec(robot, pin_root=pin_root)
    return spec.compile()


def _build_spec(robot: str, *, pin_root: bool = False) -> mujoco.MjSpec:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if robot == "g1":
        xml = MJLAB_ROOT / "src" / "mjlab" / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
        actuators = G1_ACTUATORS
        pose = G1_STANDING_POSE
        trunk_z: float | None = G1_TRUNK_Z
        timestep = 0.002
        ground_z = 0.0
    elif robot == "go1":
        xml = MJLAB_ROOT / "src" / "mjlab" / "asset_zoo" / "robots" / "unitree_go1" / "xmls" / "go1.xml"
        actuators = GO1_ACTUATORS
        pose = GO1_STANDING_POSE
        trunk_z = GO1_TRUNK_Z
        timestep = 0.005
        ground_z = 0.0
    elif robot == "yam":
        xml = MJLAB_ROOT / "src" / "mjlab" / "asset_zoo" / "robots" / "i2rt_yam" / "xmls" / "yam.xml"
        actuators = YAM_ACTUATORS
        pose = YAM_STANDING_POSE
        trunk_z = None  # fixed-base arm
        timestep = 0.002
        ground_z = -0.5  # tabletop arm; floor below the base
    else:
        raise ValueError(f"unknown robot: {robot}")

    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = timestep
    spec.option.iterations = 8
    spec.option.ls_iterations = 8
    _add_floor(spec, ground_z)
    _add_position_actuators(spec, actuators)
    if pin_root:
        _weld_root_to_world(spec, trunk_z)
    actuator_order = [a[0] for a in actuators]
    _add_keyframe(spec, "stand", trunk_z if not pin_root else None, pose, actuator_order)
    return spec


def build_scene(robot: str, *, pin_root: bool = False) -> Path:
    """Build and serialize the scene to disk; mostly useful for external tools.

    Note: the saved XML preserves the include + relative meshdir from the
    robot's own MJCF, so it'll only load correctly if the running process'
    cwd lets mujoco resolve those paths. The in-memory `build_compiled_model`
    is the canonical entry point.
    """
    spec = _build_spec(robot, pin_root=pin_root)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_pendant" if pin_root else "_freefall"
    out = CACHE_DIR / f"{robot}{suffix}_scene.xml"
    out.write_text(spec.to_xml())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["g1", "go1", "yam", "all"], default="all")
    p.add_argument(
        "--variant",
        choices=["pendant", "freefall", "both"],
        default="both",
        help="pendant=root welded to world (holds in air); freefall=normal floating-base",
    )
    args = p.parse_args(argv)
    robots = ["g1", "go1", "yam"] if args.robot == "all" else [args.robot]
    variants = ["pendant", "freefall"] if args.variant == "both" else [args.variant]
    for r in robots:
        for v in variants:
            path = build_scene(r, pin_root=(v == "pendant"))
            print(f"[build_stand_scenes] {r:>4s} {v:>9s} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
