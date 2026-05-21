"""Runtime MJCF patching to add ``<position>`` actuators (mjlab parity helper).

mjlab adds actuators programmatically via ``mujoco.MjSpec`` so each joint gets
PD-controlled position actuation via MuJoCo's semi-implicit integrator
(unconditionally stable vs explicit Python PD which destabilizes at the
gains we need). This helper does the same and writes the patched XML to
a tempfile that MetaSim's ``SceneCfg(mjcf_path=...)`` can load.

Usage:
    from roboverse_pack.tasks.mjlab._mjcf_patch import patch_mjcf_with_pd_actuators
    patched_xml = patch_mjcf_with_pd_actuators(
        "/path/to/robot.xml",
        joint_kp={"FR_hip_joint": 15.9, ...},
    )
    scenario.scene.mjcf_path = patched_xml
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import mujoco


def patch_mjcf_with_pd_actuators(
    mjcf_path: str,
    joint_kp: dict[str, float],
    damping_ratio: float = 2.0,
    natural_freq_hz: float = 10.0,
    cache: bool = True,
    integrator: str = "implicitfast",
    solver_iterations: int = 100,
    solver_ls_iterations: int = 50,
    timestep: float | None = None,
) -> str:
    """Add ``<position>`` actuators to each named joint, return path to patched XML.

    Also sets MuJoCo solver options to mjlab defaults — most critically
    ``integrator="implicitfast"`` which keeps stiff PD setups stable
    (explicit Euler diverges on G1's default pose by step 2 even with
    zero action; mjlab uses ``implicitfast`` everywhere for this reason).

    Args:
        mjcf_path: Path to source MJCF.
        joint_kp: Per-joint stiffness in N·m/rad. Each named joint gets one
            position actuator with ``gainprm=[kp]`` and ``biasprm=[0,-kp,-kv]``
            where ``kv = 2*ζ*kp/(2π*f)`` (critically damped scaled by ratio).
        damping_ratio: Critical-damping multiplier (mjlab default 2.0).
        natural_freq_hz: Used to derive kv. mjlab default 10 Hz.
        cache: If True (default), reuse patched XML keyed by md5 of inputs.
        integrator: MuJoCo integrator — ``"implicitfast"`` (mjlab default)
            or ``"euler"``. ``"implicitfast"`` is the only one stable for
            stiff humanoid PD at dt ≥ 0.002.
        solver_iterations: ``model.opt.iterations`` (mjlab default 100).
        solver_ls_iterations: ``model.opt.ls_iterations`` (mjlab default 50).
        timestep: Override ``model.opt.timestep``. If None, leaves the
            MJCF's own value alone.

    Returns:
        Absolute path to the patched MJCF.
    """
    key = hashlib.md5(
        (
            f"{mjcf_path}|{sorted(joint_kp.items())}|{damping_ratio}|{natural_freq_hz}"
            f"|{integrator}|{solver_iterations}|{solver_ls_iterations}|{timestep}"
        ).encode()
    ).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"mjlab_patched_{key}.xml")
    if cache and os.path.exists(out_path):
        return out_path

    spec = mujoco.MjSpec.from_file(mjcf_path)
    # Patched XML lives in /tmp; original mesh dir is relative to mjcf_path's
    # directory ("assets" subfolder). Re-anchor meshdir to absolute path so
    # MetaSim loader (which reads patched_xml from /tmp) finds the meshes.
    orig_dir = os.path.dirname(os.path.abspath(mjcf_path))
    if spec.meshdir:
        spec.meshdir = os.path.join(orig_dir, spec.meshdir)
    else:
        spec.meshdir = orig_dir

    # mjlab parity: set integrator + solver iterations on the MjSpec.
    _INTEGRATOR_MAP = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    spec.option.integrator = _INTEGRATOR_MAP[integrator]
    spec.option.iterations = int(solver_iterations)
    spec.option.ls_iterations = int(solver_ls_iterations)
    if timestep is not None:
        spec.option.timestep = float(timestep)

    omega = 2.0 * 3.14159265358979 * natural_freq_hz

    for joint in spec.joints:
        if joint.name not in joint_kp:
            continue
        kp = joint_kp[joint.name]
        kv = 2.0 * damping_ratio * kp / omega
        # Use joint's own range as ctrlrange so policy can target full range
        jr = joint.range  # tuple/list of (lo, hi)
        if jr is not None and len(jr) == 2 and (jr[0] != jr[1]):
            ctrl_lo, ctrl_hi = float(jr[0]), float(jr[1])
        else:
            ctrl_lo, ctrl_hi = -3.14, 3.14

        act = spec.add_actuator()
        act.name = f"{joint.name}_pos"
        act.target = joint.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        act.biasprm = [0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0]
        act.ctrlrange = [ctrl_lo, ctrl_hi]
        act.ctrllimited = True

    spec.compile()  # validate before writing
    with open(out_path, "w") as f:
        f.write(spec.to_xml())
    return out_path


# ---------------------------------------------------------------------------
# Robot-specific KP maps (matched to mjlab go1_constants.py / g1_constants.py)
# ---------------------------------------------------------------------------

# Go1 hip/thigh share STIFFNESS_HIP, calf uses STIFFNESS_KNEE
# (reflected_inertia × NATURAL_FREQ² from mjlab/asset_zoo/robots/unitree_go1/go1_constants.py)
# YAM 6-DOF arm — conservative kp (explicit Euler dt=0.002 stability)
YAM_KP = {
    "joint1": 20.0,
    "joint2": 20.0,
    "joint3": 15.0,
    "joint4": 10.0,
    "joint5": 10.0,
    "joint6": 8.0,
}

GO1_KP = {
    # hip+thigh: kp = 0.000111842 * 36 * (2π*10)² ≈ 15.9
    "FR_hip_joint": 15.9,
    "FR_thigh_joint": 15.9,
    "FR_calf_joint": 35.8,
    "FL_hip_joint": 15.9,
    "FL_thigh_joint": 15.9,
    "FL_calf_joint": 35.8,
    "RR_hip_joint": 15.9,
    "RR_thigh_joint": 15.9,
    "RR_calf_joint": 35.8,
    "RL_hip_joint": 15.9,
    "RL_thigh_joint": 15.9,
    "RL_calf_joint": 35.8,
}


def patch_mjcf_add_cube_and_table(
    mjcf_path: str,
    cube_pos: tuple[float, float, float] = (0.3, 0.0, 0.05),
    cube_size: float = 0.025,
    table_pos: tuple[float, float, float] = (0.3, 0.0, 0.0),
    table_size: tuple[float, float, float] = (0.3, 0.3, 0.02),
    cache: bool = True,
) -> str:
    """Add a cube (free joint) + a static table to an MJCF for manipulation tasks.

    Used for lift_cube_yam_v2: yam.xml has just the arm; we inject the cube
    body programmatically so the policy has something to reach for. Cube is
    a free joint so it can be picked up; table is a static box.
    """
    import hashlib
    import os
    import tempfile

    import mujoco

    key = hashlib.md5(f"{mjcf_path}|cube{cube_pos}|table{table_pos}|sz{cube_size}".encode()).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"mjlab_with_cube_{key}.xml")
    if cache and os.path.exists(out_path):
        return out_path

    spec = mujoco.MjSpec.from_file(mjcf_path)
    orig_dir = os.path.dirname(os.path.abspath(mjcf_path))
    if spec.meshdir:
        spec.meshdir = os.path.join(orig_dir, spec.meshdir)
    else:
        spec.meshdir = orig_dir

    # Add a table (static box geom on the world)
    world = spec.worldbody
    table = world.add_body(name="table", pos=table_pos)
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(table_size), rgba=[0.6, 0.4, 0.2, 1])

    # Add a cube as a free body
    cube_body = world.add_body(name="cube", pos=cube_pos)
    cube_body.add_freejoint(name="cube_freejoint")
    cube_body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube_size, cube_size, cube_size],
        rgba=[1.0, 0.3, 0.1, 1],
        mass=0.05,
    )

    spec.compile()
    with open(out_path, "w") as f:
        f.write(spec.to_xml())
    return out_path


# G1 (Unitree humanoid) PD gains — conservative for explicit-Euler stability.
# Real mjlab uses much higher (waist 200, leg 100) with semi-implicit integrator;
# we cap at ~30 for stability under explicit Euler @ dt=0.002.
G1_KP = {
    **{f"{side}_hip_{axis}_joint": 30.0 for side in ("left", "right") for axis in ("pitch", "roll", "yaw")},
    **{f"{side}_knee_joint": 30.0 for side in ("left", "right")},
    **{f"{side}_ankle_{axis}_joint": 15.0 for side in ("left", "right") for axis in ("pitch", "roll")},
    "waist_yaw_joint": 30.0,
    "waist_roll_joint": 30.0,
    "waist_pitch_joint": 30.0,
    **{f"{side}_shoulder_{axis}_joint": 15.0 for side in ("left", "right") for axis in ("pitch", "roll", "yaw")},
    **{f"{side}_elbow_joint": 15.0 for side in ("left", "right")},
    **{f"{side}_wrist_{axis}_joint": 8.0 for side in ("left", "right") for axis in ("roll", "pitch", "yaw")},
}
