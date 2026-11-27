"""Scripted manipulation skills on the benchmark's policy contract: an oracle baseline and an agent's tool set.

A skill turns a Cartesian intent (move the tool centre point here, close the gripper) into the joint
targets the harness steps, using only what the policy is given: the robot config from the context
(its MJCF, for kinematics) and the observation (joint positions, object poses). Nothing here reaches
into the simulator. :class:`ScriptedPolicy` chains skills from a per-task plan built on the first
observation of every episode; it is the privileged "knows the poses" ceiling a learned policy is
measured against, and :mod:`metasim.bench.agent` exposes the same skills to a language model.

Kinematics come from the robot's own MJCF through MuJoCo's model (forward kinematics and point
Jacobians), so the skills work on any backend that runs the same robot description.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policy import PolicyContext, _hold_dict_action

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------------------------- maths


def _quat_to_mat(q) -> np.ndarray:
    """Rotation matrix of a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rot_vec(rot: np.ndarray) -> np.ndarray:
    """Rotation vector (axis x angle) of a rotation matrix."""
    cos = (np.trace(rot) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, cos)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:  # half turn: the axis is the eigenvector for +1
        w, v = np.linalg.eigh(rot)
        axis = v[:, np.argmax(w)]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` radians about ``axis`` (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def top_down_rot(yaw: float) -> np.ndarray:
    """Tool orientation for a top-down grasp: tool z pointing at the table, fingers closing across ``yaw``."""
    return _rot_about(np.array([0.0, 0.0, 1.0]), yaw) @ _rot_about(np.array([1.0, 0.0, 0.0]), math.pi)


def object_yaw(quat) -> float:
    """Yaw of an object's x axis in the world."""
    rot = _quat_to_mat(quat)
    return math.atan2(rot[1, 0], rot[0, 0])


def wrap_symmetric(yaw: float, period: float = math.pi / 2) -> float:
    """``yaw`` reduced to ``(-period/2, period/2]`` (a cube looks the same every quarter turn)."""
    return yaw - round(yaw / period) * period


# ------------------------------------------------------------------------------------------ the robot


class ArmModel:
    """Forward kinematics and damped least-squares IK of one robot, from its MJCF, in the robot base frame.

    Args:
        robot_cfg: The robot config (``mjcf_path``, ``ee_body_name``, ``ee_joint_names``, ``joint_limits``).
        tcp_offset: Tool centre point in the end-effector body frame, metres (Franka: fingertip centre).
    """

    def __init__(self, robot_cfg: Any, *, tcp_offset: Vec3 = (0.0, 0.0, 0.10312)) -> None:
        import mujoco

        from metasim.utils.hf_util import check_and_download_single

        path = robot_cfg.mjcf_path
        check_and_download_single(path)
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_cfg.ee_body_name)
        if self.ee_body < 0:
            raise ValueError(f"{robot_cfg.name}: {path} has no body {robot_cfg.ee_body_name!r}")
        ee_joints = set(getattr(robot_cfg, "ee_joint_names", []) or [])
        self.joint_names: list[str] = sorted(robot_cfg.actuators)  # the handler's action order
        self.arm_joints: list[str] = [j for j in self.joint_names if j not in ee_joints]
        self.gripper_joints: list[str] = [j for j in self.joint_names if j in ee_joints]
        self.qpos_adr = np.array([self.model.joint(j).qposadr[0] for j in self.arm_joints])
        self.dof_adr = np.array([self.model.joint(j).dofadr[0] for j in self.arm_joints])
        limits = robot_cfg.joint_limits or {}
        self.low = np.array([limits.get(j, (-np.inf, np.inf))[0] for j in self.arm_joints])
        self.high = np.array([limits.get(j, (-np.inf, np.inf))[1] for j in self.arm_joints])
        self.gripper_open = float(max(limits.get(j, (0.0, 0.04))[1] for j in self.gripper_joints)) if ee_joints else 0.0
        self.gripper_closed = (
            float(min(limits.get(j, (0.0, 0.04))[0] for j in self.gripper_joints)) if ee_joints else 0.0
        )
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(tcp position, tcp rotation)`` in the base frame for the arm joint vector ``q_arm``."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_adr] = q_arm
        self._mujoco.mj_kinematics(self.model, self.data)
        self._mujoco.mj_comPos(self.model, self.data)  # Jacobians need the body COM frames
        rot = self.data.xmat[self.ee_body].reshape(3, 3).copy()
        pos = self.data.xpos[self.ee_body] + rot @ self.tcp_offset
        return pos, rot

    def _jacobian(self) -> np.ndarray:
        """6 x n_arm Jacobian of the tool centre point (translation rows first) at the current ``data``."""
        rot = self.data.xmat[self.ee_body].reshape(3, 3)
        point = self.data.xpos[self.ee_body] + rot @ self.tcp_offset
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.ee_body)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

    def ik(
        self,
        q0: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None,
        *,
        iters: int = 40,
        damping: float = 1e-2,
        max_step: float = 0.2,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, float, float]:
        """Joint vector reaching ``target_pos`` (and ``target_rot``) from ``q0``; returns ``(q, pos_err, rot_err)``.

        Damped least squares with the step clipped to ``max_step`` radians per joint and the joints clamped
        to their limits; a target out of reach returns the closest pose found.
        """
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.low, self.high)
        pos_err = rot_err = float("inf")
        for _ in range(iters):
            pos, rot = self.fk(q)
            e_pos = target_pos - pos
            e_rot = _rot_vec(target_rot @ rot.T) if target_rot is not None else np.zeros(3)
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            if pos_err < tol and (target_rot is None or rot_err < 1e-3):
                break
            jac = self._jacobian()
            if target_rot is None:
                jac, err = jac[:3], e_pos
            else:
                err = np.concatenate([e_pos, 0.5 * e_rot])
            jjt = jac @ jac.T
            dq = jac.T @ np.linalg.solve(jjt + damping * np.eye(jjt.shape[0]), err)
            scale = max_step / max(float(np.max(np.abs(dq))), 1e-12)
            if scale < 1.0:
                dq = dq * scale
            q = np.clip(q + dq, self.low, self.high)
        return q, pos_err, rot_err


# ------------------------------------------------------------------------------------- the observation


@dataclass
class EnvView:
    """One env's slice of a ``TensorState``, as numpy: joint positions, base pose and object poses."""

    robot: str
    q: dict[str, float]
    """Joint name -> position, every joint the observation reports."""
    base_pos: np.ndarray
    base_rot: np.ndarray
    objects: dict[str, tuple[np.ndarray, np.ndarray]]
    """Object name -> ``(position, (w, x, y, z) quaternion)`` in the world."""

    def world_to_base(self, pos: np.ndarray, rot: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """A world pose expressed in the robot base frame."""
        p = self.base_rot.T @ (np.asarray(pos, dtype=float) - self.base_pos)
        return p, (self.base_rot.T @ rot if rot is not None else None)

    def base_to_world(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A base-frame pose expressed in the world."""
        return self.base_rot @ pos + self.base_pos, self.base_rot @ rot

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(position, quaternion)`` of object ``name``; a missing object names the ones observed."""
        try:
            return self.objects[name]
        except KeyError:
            raise KeyError(f"object {name!r} is not observed; objects: {sorted(self.objects)}") from None


def env_view(obs: Any, env_id: int, robot: str, joint_names: list[str]) -> EnvView:
    """The :class:`EnvView` of env ``env_id`` from a ``TensorState``-shaped observation.

    Needs the robot's joint positions (proprioception); object poses are whatever the tier left in
    ``obs.objects`` (none below privileged).
    """
    robots = getattr(obs, "robots", None) or {}
    if robot not in robots:
        raise ValueError(f"the observation carries no state for robot {robot!r} (proprioception off?)")
    rs = robots[robot]
    joint_pos = rs.joint_pos[env_id].detach().cpu().numpy().astype(float)
    if joint_pos.shape[0] != len(joint_names):
        raise ValueError(f"{robot}: {joint_pos.shape[0]} joint positions for {len(joint_names)} joints")
    root = rs.root_state[env_id].detach().cpu().numpy().astype(float)
    objects = {}
    for name, st in (getattr(obs, "objects", None) or {}).items():
        row = st.root_state[env_id].detach().cpu().numpy().astype(float)
        objects[name] = (row[:3], row[3:7])
    return EnvView(
        robot=robot,
        q=dict(zip(joint_names, joint_pos.tolist(), strict=True)),
        base_pos=root[:3],
        base_rot=_quat_to_mat(root[3:7]),
        objects=objects,
    )


# ------------------------------------------------------------------------------------------- the skills


class Skill:
    """One closed-loop primitive; ``step`` returns ``(arm joint targets, gripper opening, done)``."""

    name = "skill"

    def start(self, view: EnvView, arm: ArmModel) -> None:
        """Latch the target from the first observation the skill runs on."""

    def step(self, view: EnvView, arm: ArmModel) -> tuple[np.ndarray, float, bool]:
        """Targets for this step and whether the skill is complete."""
        raise NotImplementedError

    def describe(self) -> str:
        """One line for logs and agent transcripts."""
        return self.name


PoseFn = Callable[[EnvView], tuple[np.ndarray, np.ndarray | None]]


@dataclass
class MoveTo(Skill):
    """Move the tool centre point to a world pose along a straight line, at most ``speed`` metres per step.

    ``target`` is a ``(position, rotation matrix or None)`` pair or a function of the first observation
    producing one (so a plan can name "above the cube" before the cube has been seen). The gripper is
    held at ``gripper`` metres. Done when within ``tol`` metres and ``rot_tol`` radians, or after
    ``max_steps`` (recorded in ``stalled``).
    """

    target: PoseFn | tuple[np.ndarray, np.ndarray | None]
    gripper: float
    speed: float = 0.03
    rot_speed: float = 0.25
    tol: float = 0.01
    rot_tol: float = 0.15
    max_steps: int = 150
    name: str = "move_to"
    _goal: tuple[np.ndarray, np.ndarray | None] | None = field(default=None, repr=False)
    _steps: int = field(default=0, repr=False)
    stalled: bool = field(default=False, repr=False)

    def start(self, view: EnvView, arm: ArmModel) -> None:
        """Resolve the target pose from the first observation."""
        goal = self.target(view) if callable(self.target) else self.target
        pos, rot = goal
        self._goal = (np.asarray(pos, dtype=float), None if rot is None else np.asarray(rot, dtype=float))
        self._steps = 0
        self.stalled = False

    def step(self, view: EnvView, arm: ArmModel) -> tuple[np.ndarray, float, bool]:
        """Advance one waypoint towards the goal and solve the joints for it."""
        assert self._goal is not None, "start() first"
        q = np.array([view.q[j] for j in arm.arm_joints])
        cur_pos_b, cur_rot_b = arm.fk(q)
        cur_pos, cur_rot = view.base_to_world(cur_pos_b, cur_rot_b)
        goal_pos, goal_rot = self._goal
        delta = goal_pos - cur_pos
        dist = float(np.linalg.norm(delta))
        way_pos = goal_pos if dist <= self.speed else cur_pos + delta / dist * self.speed
        rot_err = 0.0
        way_rot = None
        if goal_rot is not None:
            rv = _rot_vec(goal_rot @ cur_rot.T)
            rot_err = float(np.linalg.norm(rv))
            way_rot = goal_rot if rot_err <= self.rot_speed else _rot_about(rv, self.rot_speed) @ cur_rot
        p_b, r_b = view.world_to_base(way_pos, way_rot)
        q_cmd, _, _ = arm.ik(q, p_b, r_b, iters=15)
        self._steps += 1
        done = dist < self.tol and rot_err < self.rot_tol
        if not done and self._steps >= self.max_steps:
            self.stalled = True
            done = True
        return q_cmd, self.gripper, done


@dataclass
class Gripper(Skill):
    """Open or close the gripper to ``opening`` metres while holding the arm, for ``steps`` steps."""

    opening: float
    steps: int = 12
    name: str = "gripper"
    _q: np.ndarray | None = field(default=None, repr=False)
    _steps: int = field(default=0, repr=False)

    def start(self, view: EnvView, arm: ArmModel) -> None:
        """Latch the arm pose to hold."""
        self._q = np.array([view.q[j] for j in arm.arm_joints])
        self._steps = 0

    def step(self, view: EnvView, arm: ArmModel) -> tuple[np.ndarray, float, bool]:
        """Hold the arm, command the opening; done after ``steps``."""
        assert self._q is not None, "start() first"
        self._steps += 1
        return self._q, self.opening, self._steps >= self.steps


@dataclass
class MoveJoints(Skill):
    """Interpolate the arm in joint space to the configuration reaching a world pose, over ``steps`` steps.

    The target configuration is solved once at the start (a long IK run from the current joints), so a
    large reorientation that a Cartesian straight line would drag through singularities moves as one
    smooth joint motion. ``target`` is as for :class:`MoveTo`.
    """

    target: PoseFn | tuple[np.ndarray, np.ndarray | None]
    gripper: float
    steps: int = 40
    name: str = "move_joints"
    _q0: np.ndarray | None = field(default=None, repr=False)
    _q1: np.ndarray | None = field(default=None, repr=False)
    _steps: int = field(default=0, repr=False)
    stalled: bool = field(default=False, repr=False)

    def start(self, view: EnvView, arm: ArmModel) -> None:
        """Solve the goal configuration from the current joints."""
        pos, rot = self.target(view) if callable(self.target) else self.target
        p_b, r_b = view.world_to_base(
            np.asarray(pos, dtype=float), None if rot is None else np.asarray(rot, dtype=float)
        )
        self._q0 = np.array([view.q[j] for j in arm.arm_joints])
        self._q1, pos_err, rot_err = arm.ik(self._q0, p_b, r_b, iters=300, max_step=0.1)
        self.stalled = pos_err > 0.02 or rot_err > 0.2  # the pose is out of reach; go as close as the solver got
        self._steps = 0

    def step(self, view: EnvView, arm: ArmModel) -> tuple[np.ndarray, float, bool]:
        """The next point of the joint-space line."""
        assert self._q0 is not None and self._q1 is not None, "start() first"
        self._steps += 1
        a = min(1.0, self._steps / self.steps)
        return (1 - a) * self._q0 + a * self._q1, self.gripper, self._steps >= self.steps


# -------------------------------------------------------------------------------------- plan building


def above(name: str, dz: float = 0.12, yaw: float | None = None) -> PoseFn:
    """Pose ``dz`` metres above object ``name`` with a top-down tool, fingers across the object's faces."""

    def fn(view: EnvView):
        pos, quat = view.pose(name)
        y = wrap_symmetric(object_yaw(quat)) if yaw is None else yaw
        return pos + np.array([0.0, 0.0, dz]), top_down_rot(y)

    return fn


def at(name: str, dz: float = 0.0, yaw: float | None = None) -> PoseFn:
    """Pose at object ``name``'s centre (plus ``dz``) with a top-down tool."""
    return above(name, dz, yaw)


def pick(name: str, arm: ArmModel, *, lift: float = 0.2) -> list[Skill]:
    """Top-down grasp of object ``name`` and a lift of ``lift`` metres."""
    return [
        MoveTo(above(name), arm.gripper_open, name="reach_above"),
        MoveTo(at(name), arm.gripper_open, speed=0.02, name="descend"),
        Gripper(arm.gripper_closed, name="close"),
        MoveTo(_offset_current(0.0, 0.0, lift), arm.gripper_closed, name="lift"),
    ]


def place_on(name: str, arm: ArmModel, *, clearance: float = 0.045, hover: float = 0.15) -> list[Skill]:
    """Carry the held object over ``name`` and set it down ``clearance`` metres above its centre."""
    return [
        MoveTo(above(name, hover), arm.gripper_closed, name="carry"),
        MoveTo(above(name, clearance), arm.gripper_closed, speed=0.02, name="lower"),
        Gripper(arm.gripper_open, name="release"),
        MoveTo(_offset_current(0.0, 0.0, 0.1), arm.gripper_open, name="retreat"),
    ]


def push_to(name: str, goal: str | tuple[float, float], arm: ArmModel, *, height: float = 0.02) -> list[Skill]:
    """Push object ``name`` along the table to ``goal`` (an object name or an ``(x, y)``) with closed fingers."""

    def direction(view: EnvView) -> tuple[np.ndarray, np.ndarray]:
        pos, _ = view.pose(name)
        gxy = np.asarray(view.pose(goal)[0][:2] if isinstance(goal, str) else goal, dtype=float)
        d = gxy - pos[:2]
        n = float(np.linalg.norm(d))
        return pos, (d / n if n > 1e-6 else np.array([1.0, 0.0]))

    def behind(view: EnvView):
        pos, d = direction(view)
        xy = pos[:2] - d * 0.08
        return np.array([xy[0], xy[1], height + 0.12]), top_down_rot(math.atan2(d[1], d[0]))

    def contact(view: EnvView):
        pos, d = direction(view)
        xy = pos[:2] - d * 0.07
        return np.array([xy[0], xy[1], height]), top_down_rot(math.atan2(d[1], d[0]))

    def through(view: EnvView):
        pos, d = direction(view)
        gxy = np.asarray(view.pose(goal)[0][:2] if isinstance(goal, str) else goal, dtype=float)
        xy = gxy - d * 0.025
        return np.array([xy[0], xy[1], height]), top_down_rot(math.atan2(d[1], d[0]))

    return [
        Gripper(arm.gripper_closed, steps=6, name="close"),
        MoveTo(behind, arm.gripper_closed, name="approach"),
        MoveTo(contact, arm.gripper_closed, speed=0.02, name="touch_down"),
        MoveTo(through, arm.gripper_closed, speed=0.012, tol=0.008, name="push"),
    ]


def stand_upright(name: str, arm: ArmModel, *, length: float, width: float) -> list[Skill]:
    """Grasp a lying bar ``name`` (its x axis along its length) at the centre, lift it and stand it on one end."""
    state: dict[str, Any] = {}

    def grasp_pose(view: EnvView, dz: float):
        pos, quat = view.pose(name)
        yaw = object_yaw(quat)
        if "yaw" not in state:
            state["yaw"] = yaw
        return pos + np.array([0.0, 0.0, dz]), top_down_rot(state["yaw"])

    def upright_rot() -> np.ndarray:
        yaw = state["yaw"]
        u = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # the bar's axis in the world at grasp time
        axis = np.cross(u, np.array([0.0, 0.0, 1.0]))
        return _rot_about(axis, math.pi / 2) @ top_down_rot(yaw)

    def rotate(view: EnvView):
        q = np.array([view.q[j] for j in arm.arm_joints])
        pos_b, _ = arm.fk(q)
        pos, _ = view.base_to_world(pos_b, np.eye(3))
        return pos, upright_rot()

    def set_down(view: EnvView):
        q = np.array([view.q[j] for j in arm.arm_joints])
        pos_b, _ = arm.fk(q)
        pos, _ = view.base_to_world(pos_b, np.eye(3))
        return np.array([pos[0], pos[1], length / 2 + 0.004]), upright_rot()

    def back_off(view: EnvView):
        q = np.array([view.q[j] for j in arm.arm_joints])
        pos_b, rot_b = arm.fk(q)
        pos, rot = view.base_to_world(pos_b, rot_b)
        return pos - rot[:, 2] * 0.12, rot  # straight back along the tool axis

    return [
        MoveTo(lambda v: grasp_pose(v, 0.12), arm.gripper_open, name="reach_above"),
        MoveTo(lambda v: grasp_pose(v, 0.0), arm.gripper_open, speed=0.02, name="descend"),
        Gripper(arm.gripper_closed, name="close"),
        MoveTo(_offset_current(0.0, 0.0, 0.25), arm.gripper_closed, name="lift"),
        MoveJoints(rotate, arm.gripper_closed, steps=50, name="rotate_upright"),
        MoveTo(set_down, arm.gripper_closed, speed=0.015, max_steps=80, name="set_down"),
        Gripper(arm.gripper_open, name="release"),
        MoveTo(back_off, arm.gripper_open, name="back_off"),
    ]


def _offset_current(dx: float, dy: float, dz: float) -> PoseFn:
    """The tool's current pose shifted by ``(dx, dy, dz)`` in the world, orientation kept."""

    def fn(view: EnvView):
        arm = _CURRENT_ARM.get(view.robot)
        if arm is None:
            raise RuntimeError("offset targets need the arm model (ScriptedPolicy sets it)")
        q = np.array([view.q[j] for j in arm.arm_joints])
        pos_b, rot_b = arm.fk(q)
        pos, rot = view.base_to_world(pos_b, rot_b)
        return pos + np.array([dx, dy, dz]), rot

    return fn


_CURRENT_ARM: dict[str, ArmModel] = {}
"""Robot name -> arm model of the running policy, for relative targets built before the arm existed."""


PlanFn = Callable[[EnvView, ArmModel], list[Skill]]

TABLETOP_PLANS: dict[str, PlanFn] = {
    "pick_cube": lambda v, arm: pick("cube", arm, lift=0.2),
    "stack_cube": lambda v, arm: pick("cube", arm, lift=0.12) + place_on("base", arm),
    "push_cube": lambda v, arm: push_to("cube", "goal_region", arm),
    "pull_cube": lambda v, arm: push_to("cube", "goal_region", arm),
    "lift_peg_upright": lambda v, arm: stand_upright("peg", arm, length=0.24, width=0.05),
}
"""Scripted plans for the ``tabletop-mujoco`` suite, keyed by task registry name (and matched by class name)."""


def plan_for(task: str) -> PlanFn:
    """The plan registered for ``task`` (a registry name such as ``pick_cube`` or a class name such as ``PickCubeTask``)."""
    key = task.lower()
    for suffix in ("task", "cfg", "env"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    key = key.replace("_", "").replace(".", "")
    for name, plan in TABLETOP_PLANS.items():
        if name.replace("_", "") == key or key.endswith(name.replace("_", "")):
            return plan
    raise KeyError(f"no scripted plan for task {task!r}; plans: {sorted(TABLETOP_PLANS)}")


# ------------------------------------------------------------------------------------------- the policy


class SkillRunner:
    """Runs a list of skills to completion for one env; returns dict actions for the harness."""

    def __init__(self, arm: ArmModel, robot: str) -> None:
        self.arm = arm
        self.robot = robot
        self.skills: list[Skill] = []
        self.index = 0
        self.log: list[str] = []
        self._started = False

    def load(self, skills: list[Skill]) -> None:
        """Replace the plan; the first skill starts on the next ``act``."""
        self.skills = list(skills)
        self.index = 0
        self._started = False

    @property
    def done(self) -> bool:
        """Every skill has finished."""
        return self.index >= len(self.skills)

    @property
    def current(self) -> Skill | None:
        """The skill in progress, if any."""
        return None if self.done else self.skills[self.index]

    def act(self, view: EnvView) -> dict:
        """One step of the current skill (the hold action once the plan is exhausted)."""
        q_arm = np.array([view.q[j] for j in self.arm.arm_joints])
        gripper = float(view.q[self.arm.gripper_joints[0]]) if self.arm.gripper_joints else 0.0
        while not self.done:
            skill = self.skills[self.index]
            if not self._started:
                skill.start(view, self.arm)
                self._started = True
            q_arm, gripper, finished = skill.step(view, self.arm)
            if finished:
                stalled = getattr(skill, "stalled", False)
                self.log.append(f"{skill.name}{' (stalled)' if stalled else ''}")
                self.index += 1
                self._started = False
            break
        targets = dict(zip(self.arm.arm_joints, (float(v) for v in q_arm), strict=True))
        for j in self.arm.gripper_joints:
            targets[j] = gripper
        return {self.robot: {"dof_pos_target": targets}}


class ScriptedPolicy:
    """The scripted-skills baseline: a per-task plan over privileged object poses, executed closed-loop.

    Args:
        task: Plan to use (registry or class name); default: the task named by the context.
        tcp_offset: Tool centre point in the end-effector frame (Franka default).

    Needs object poses in the observation, so it runs under the ``privileged`` tier only; below it the
    first observation raises. ``family = "scripted"`` is recorded in the result.
    """

    name = "scripted"
    family = "scripted"

    def __init__(self, task: str | None = None, tcp_offset: Vec3 = (0.0, 0.0, 0.10312)) -> None:
        self._task = task
        self._tcp = tuple(float(v) for v in tcp_offset)
        self._arm: ArmModel | None = None
        self._robot: str | None = None
        self._runners: dict[int, SkillRunner] = {}
        self._pending: set[int] = set()
        self.episode_logs: list[list[str]] = []

    def _setup(self, ctx: PolicyContext) -> None:
        if self._arm is not None:
            return
        scenario = ctx.extras.get("scenario")
        robots = list(getattr(scenario, "robots", []) or [])
        if len(robots) != 1:
            raise ValueError(f"ScriptedPolicy drives one robot; the scenario has {len(robots)}")
        self._robot = robots[0].name
        self._arm = ArmModel(robots[0], tcp_offset=self._tcp)
        _CURRENT_ARM[self._robot] = self._arm
        joints = ctx.action_joint_names[self._robot]
        if list(joints) != self._arm.joint_names:
            raise ValueError(f"action joint order {joints} differs from the robot config's {self._arm.joint_names}")

    def reset(self, env_ids: list[int], instances, ctx: PolicyContext) -> None:
        """Plans are built on the first observation of each episode (poses are state, not config)."""
        self._setup(ctx)
        for env_id in env_ids:
            runner = self._runners.get(env_id)
            if runner is not None and runner.log:
                self.episode_logs.append(runner.log)
            self._runners[env_id] = SkillRunner(self._arm, self._robot)
            self._pending.add(env_id)

    def act(self, obs: Any, ctx: PolicyContext) -> list:
        """Dict actions for every env slot; idle slots hold."""
        self._setup(ctx)
        batch = []
        hold = None
        for env_id in range(ctx.num_envs):
            runner = self._runners.get(env_id)
            if runner is None:
                hold = hold or _hold_dict_action(ctx)
                batch.append(hold)
                continue
            view = env_view(obs, env_id, self._robot, self._arm.joint_names)
            if env_id in self._pending:
                self._pending.discard(env_id)
                runner.load(plan_for(self._task or ctx.task)(view, self._arm))
            batch.append(runner.act(view))
        return batch
