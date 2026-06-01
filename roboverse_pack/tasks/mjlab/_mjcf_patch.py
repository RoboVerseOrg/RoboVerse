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
import json
import os
import tempfile

import mujoco

# Path to the committed exact mjlab actuator spec (per-joint kp/kd/armature/
# ctrlrange/forcerange/frictionloss + integrator/timestep/solver-iterations),
# dumped 1:1 from mjlab's COMPILED g1/go1 models. See
# ``patch_mjcf_with_exact_spec`` below for how it is applied.
_ACTUATOR_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mjlab_actuator_spec.json")

# mjlab ``MujocoCfg.integrator`` values map to MuJoCo's ``mjtIntegrator`` enum.
# The spec JSON stores the raw enum int (g1/go1 use ``implicitfast`` == 3).
_INTEGRATOR_BY_INT = {
    0: mujoco.mjtIntegrator.mjINT_EULER,
    1: mujoco.mjtIntegrator.mjINT_RK4,
    2: mujoco.mjtIntegrator.mjINT_IMPLICIT,
    3: mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


def _load_actuator_spec(robot: str) -> dict:
    """Load the committed per-joint actuator spec for ``g1`` or ``go1``."""
    with open(_ACTUATOR_SPEC_PATH) as f:
        spec = json.load(f)
    if robot not in spec:
        raise KeyError(f"_mjlab_actuator_spec.json has no entry for robot '{robot}' (have {list(spec)})")
    return spec[robot]


def patch_mjcf_with_exact_spec(
    mjcf_path: str,
    robot: str,
    *,
    cache: bool = True,
) -> str:
    """Apply mjlab's EXACT compiled actuator/dynamics spec to an MJCF.

    This reproduces, on the raw asset_zoo MJCF, the per-joint actuator + joint
    dynamics that mjlab's :class:`BuiltinPositionActuatorCfg` pipeline bakes into
    the COMPILED model (verified bitwise against ``Entity(...).spec.compile()``).
    For every actuated joint named in ``_mjlab_actuator_spec.json`` it:

    * adds a single ``<position>`` actuator with ``gainprm=[kp]`` and
      ``biasprm=[0, -kp, -kd]`` (mjlab's PD-as-native-MuJoCo formulation, so the
      ``implicitfast`` integrator folds kp/kd into the velocity update and stays
      stable at the stiff leg gains — kp up to ~99 — that the policy needs);
    * sets ``forcelimited=True`` + ``forcerange=[-F, F]`` (the motor effort
      limit) and an informational ``ctrlrange`` but leaves ``ctrllimited=False``
      and ``inheritrange=0`` — mjlab deliberately does NOT clamp the position
      setpoint to the joint range, so the policy can command targets beyond the
      kinematic limit (``create_position_actuator`` in mjlab/utils/spec.py);
    * overrides the joint's ``armature`` and ``frictionloss`` (armature is part
      of mjlab's dynamics AND what lets ``implicitfast`` stay stable at high kp).

    Solver options (integrator/timestep/iterations/ls_iterations + the velocity
    env's solver=newton, cone=pyramidal, impratio=1.0) are taken from the spec /
    mjlab's ``MujocoCfg`` so the MuJoCo backend matches mjlab's native rollout.

    Args:
        mjcf_path: Path to the source asset_zoo MJCF (e.g. ``g1.xml``).
        robot: ``"g1"`` or ``"go1"`` — selects the spec block.
        cache: Reuse a previously-patched XML keyed on inputs + spec mtime.

    Returns:
        Absolute path to the patched MJCF.
    """
    spec_mtime = os.path.getmtime(_ACTUATOR_SPEC_PATH)
    key = hashlib.md5(f"exact|{mjcf_path}|{robot}|{spec_mtime}".encode()).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"mjlab_exact_{robot}_{key}.xml")
    if cache and os.path.exists(out_path):
        return out_path

    rspec = _load_actuator_spec(robot)
    joints_spec: dict[str, dict] = rspec["joints"]

    mspec = mujoco.MjSpec.from_file(mjcf_path)
    # Patched XML lives in /tmp; re-anchor meshdir so MetaSim's loader (which
    # reads the patched file from /tmp) still finds the mesh assets.
    orig_dir = os.path.dirname(os.path.abspath(mjcf_path))
    mspec.meshdir = os.path.join(orig_dir, mspec.meshdir) if mspec.meshdir else orig_dir

    # mjlab velocity-env solver options (MujocoCfg + the velocity_env_cfg sim
    # block): implicitfast / newton / pyramidal-cone / impratio=1 + the spec's
    # timestep + (ls_)iterations. gravity stays the MuJoCo default (-9.81).
    mspec.option.integrator = _INTEGRATOR_BY_INT[int(rspec["integrator"])]
    mspec.option.timestep = float(rspec["timestep"])
    mspec.option.iterations = int(rspec["iterations"])
    mspec.option.ls_iterations = int(rspec["ls_iterations"])
    mspec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mspec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    mspec.option.impratio = 1.0

    # Override per-joint dynamics (armature + frictionloss). mjlab applies these
    # via ``apply_target_overrides`` on the joint, NOT on the actuator.
    for joint in mspec.joints:
        js = joints_spec.get(joint.name)
        if js is None:
            continue
        joint.armature = float(js["armature"])
        joint.frictionloss = float(js["frictionloss"])

    # Override per-geom collision params to mjlab's compiled CollisionCfg values
    # (condim / priority / solref / solimp / friction / contype / conaffinity).
    # This matters for contact-driven locomotion: mjlab gives the feet condim=6
    # (go1) / condim=3 (g1) with priority + a specific friction cone and a
    # stiffer solref; the raw asset_zoo MJCFs leave the default condim=3,
    # solref=(0.02,1), so without this the quadruped's feet roll/slip and the
    # robot drifts forward relative to mjlab. Geom names match the compiled
    # model exactly (``*_collision``).
    collisions_spec: dict[str, dict] = rspec.get("collisions", {})
    for geom in mspec.geoms:
        gs = collisions_spec.get(geom.name)
        if gs is None:
            continue
        geom.condim = int(gs["condim"])
        geom.priority = int(gs["priority"])
        geom.contype = int(gs["contype"])
        geom.conaffinity = int(gs["conaffinity"])
        geom.solref[:] = list(gs["solref"])
        geom.solimp[:] = list(gs["solimp"])
        fr = list(gs["friction"])
        geom.friction[: len(fr)] = fr

    # One <position> actuator per actuated joint, in the spec's joint order.
    for jname, js in joints_spec.items():
        kp = float(js["kp"])
        kd = float(js["kd"])
        force = float(js["forcerange"][1])
        ctrl_lo, ctrl_hi = float(js["ctrlrange"][0]), float(js["ctrlrange"][1])

        act = mspec.add_actuator()
        act.name = jname
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        act.biasprm = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
        # mjlab leaves the position setpoint UNCLAMPED (policy may target beyond
        # joint range); effort is bounded by forcerange instead.
        act.inheritrange = 0.0
        act.ctrllimited = False
        act.ctrlrange = [ctrl_lo, ctrl_hi]  # informational only (ctrllimited=False)
        act.forcelimited = True
        act.forcerange = [-force, force]

    mspec.compile()  # validate before writing
    with open(out_path, "w") as f:
        f.write(mspec.to_xml())
    return out_path


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


def patch_mjcf_set_body_pos(
    mjcf_path: str,
    *,
    body_name: str,
    pos: tuple[float, float, float],
    cache: bool = True,
) -> str:
    """Set a named body's ``pos`` in an MJCF, return path to the patched XML.

    Used to replicate an init-state base offset that the upstream framework
    applies via a keyframe rather than the MJCF body pos. For the YAM the
    mjlab ``HOME_KEYFRAME`` places the base ``arm`` body at z=0.01
    (asset_zoo/robots/i2rt_yam/yam_constants.py HOME_KEYFRAME ``pos``);
    the raw yam.xml leaves it at the worldbody origin, so without this the
    whole robot (and every site world position) sits 0.01 m too low.
    """
    key = hashlib.md5(f"{mjcf_path}|{body_name}|{pos}".encode()).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"mjlab_bodypos_{key}.xml")
    if cache and os.path.exists(out_path):
        return out_path

    spec = mujoco.MjSpec.from_file(mjcf_path)
    orig_dir = os.path.dirname(os.path.abspath(mjcf_path))
    spec.meshdir = os.path.join(orig_dir, spec.meshdir) if spec.meshdir else orig_dir

    body = spec.body(body_name)
    if body is None:
        raise ValueError(f"patch_mjcf_set_body_pos: body '{body_name}' not found in {mjcf_path}")
    body.pos = list(pos)

    spec.compile()
    with open(out_path, "w") as f:
        f.write(spec.to_xml())
    return out_path


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
