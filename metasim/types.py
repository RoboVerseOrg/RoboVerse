"""This file contains the basic types for the MetaSim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import numpy as np
import torch

from metasim.utils.math import convert_camera_frame_orientation_convention

## Basic types
Dof = Dict[str, float]


@dataclass(frozen=True)
class ShapeSpec:
    """Machine-readable shape metadata for tensor annotations."""

    dims: Tuple[Union[str, int], ...]


RootStateTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", 13))]
BodyStateTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", "num_bodies", 13))]
JointStateTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", "num_joints"))]
CameraRgbTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", "H", "W", 3))]
CameraDepthTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", "H", "W"))]
CameraSegmentationTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", "H", "W"))]
CameraPosTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", 3))]
CameraQuatTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", 4))]
CameraIntrinsicsTensor = Annotated[torch.Tensor, ShapeSpec(("num_envs", 3, 3))]


## Trajectory types
class RobotAction(TypedDict, total=False):
    """Action of the robot."""

    dof_pos_target: Dof | None
    dof_vel_target: Dof | None
    dof_effort_target: Dof | None


Action = Dict[str, RobotAction]
ActionBatch = List[Action]
ActionInput = Union[ActionBatch, torch.Tensor]
CompatActionInput = Union[ActionInput, np.ndarray]


class DictObjectState(TypedDict):
    """State of the object."""

    pos: torch.Tensor
    rot: torch.Tensor
    vel: torch.Tensor | None
    ang_vel: torch.Tensor | None
    dof_pos: Dof | None
    dof_vel: Dof | None


class DictRobotState(DictObjectState):
    """State of the robot."""

    dof_pos: Dof | None
    dof_vel: Dof | None

    dof_pos_target: Dof | None
    dof_vel_target: Dof | None
    dof_torque: Dof | None


class DictEnvState(TypedDict):
    """State of the environment."""

    objects: dict[str, DictObjectState]
    robots: dict[str, DictRobotState]
    cameras: dict[str, dict[str, torch.Tensor]]
    extras: dict[str, Any]  # States of Extra information


DictStateBatch = List[DictEnvState]
StateMode = Literal["tensor", "dict"]


def _expect_tensor(name: str, value: torch.Tensor | None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    return value


def _validate_rank(name: str, value: torch.Tensor | None, rank: int) -> torch.Tensor:
    tensor = _expect_tensor(name, value)
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")
    return tensor


def _validate_last_dim(name: str, value: torch.Tensor | None, last_dim: int) -> torch.Tensor:
    tensor = _expect_tensor(name, value)
    if tensor.shape[-1] != last_dim:
        raise ValueError(f"{name} must end with dim {last_dim}, got shape {tuple(tensor.shape)}")
    return tensor


def _validate_num_envs(name: str, value: torch.Tensor | None, num_envs: int) -> torch.Tensor:
    tensor = _expect_tensor(name, value)
    if tensor.shape[0] != num_envs:
        raise ValueError(f"{name} must have num_envs={num_envs}, got shape {tuple(tensor.shape)}")
    return tensor


def _maybe_num_envs_from_camera_state(state: CameraState) -> int | None:
    for tensor in (
        state.rgb,
        state.depth,
        state.instance_id_seg,
        state.instance_seg,
        state.pos,
        state.quat_world,
        state.intrinsics,
    ):
        if isinstance(tensor, torch.Tensor):
            return tensor.shape[0]
    return None


@dataclass
class ObjectState:
    """State of a single object."""

    root_state: RootStateTensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str] | None = None
    """Body names. This is only available for articulation objects."""
    body_state: BodyStateTensor | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
    joint_pos: JointStateTensor | None = None
    """Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
    joint_vel: JointStateTensor | None = None
    """Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""

    def __post_init__(self) -> None:
        root_state = _validate_last_dim("root_state", _validate_rank("root_state", self.root_state, 2), 13)
        num_envs = root_state.shape[0]

        if self.body_state is not None:
            body_state = _validate_last_dim("body_state", _validate_rank("body_state", self.body_state, 3), 13)
            _validate_num_envs("body_state", body_state, num_envs)
            if self.body_names is not None and body_state.shape[1] != len(self.body_names):
                raise ValueError(
                    f"body_state second dim must match body_names length {len(self.body_names)}, "
                    f"got shape {tuple(body_state.shape)}"
                )

        if self.joint_pos is not None:
            _validate_num_envs("joint_pos", _validate_rank("joint_pos", self.joint_pos, 2), num_envs)

        if self.joint_vel is not None:
            joint_vel = _validate_num_envs("joint_vel", _validate_rank("joint_vel", self.joint_vel, 2), num_envs)
            if self.joint_pos is not None and joint_vel.shape != self.joint_pos.shape:
                raise ValueError(
                    f"joint_vel must match joint_pos shape {tuple(self.joint_pos.shape)}, got {tuple(joint_vel.shape)}"
                )


@dataclass
class RobotState:
    """State of a single robot."""

    root_state: RootStateTensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str]
    """Body names."""
    body_state: BodyStateTensor
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: JointStateTensor
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: JointStateTensor
    """Joint velocities. Shape is (num_envs, num_joints)."""
    joint_pos_target: JointStateTensor | None
    """Joint positions target. Shape is (num_envs, num_joints)."""
    joint_vel_target: JointStateTensor | None
    """Joint velocities target. Shape is (num_envs, num_joints)."""
    joint_effort_target: JointStateTensor | None
    """Joint effort targets. Shape is (num_envs, num_joints)."""

    def __post_init__(self) -> None:
        root_state = _validate_last_dim("root_state", _validate_rank("root_state", self.root_state, 2), 13)
        num_envs = root_state.shape[0]

        body_state = _validate_last_dim("body_state", _validate_rank("body_state", self.body_state, 3), 13)
        _validate_num_envs("body_state", body_state, num_envs)
        if body_state.shape[1] != len(self.body_names):
            raise ValueError(
                f"body_state second dim must match body_names length {len(self.body_names)}, "
                f"got shape {tuple(body_state.shape)}"
            )

        joint_pos = _validate_num_envs("joint_pos", _validate_rank("joint_pos", self.joint_pos, 2), num_envs)
        joint_vel = _validate_num_envs("joint_vel", _validate_rank("joint_vel", self.joint_vel, 2), num_envs)
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"joint_vel must match joint_pos shape {tuple(joint_pos.shape)}, got {tuple(joint_vel.shape)}"
            )

        for name, target in (
            ("joint_pos_target", self.joint_pos_target),
            ("joint_vel_target", self.joint_vel_target),
            ("joint_effort_target", self.joint_effort_target),
        ):
            if target is None:
                continue
            target_tensor = _validate_num_envs(name, _validate_rank(name, target, 2), num_envs)
            if target_tensor.shape != joint_pos.shape:
                raise ValueError(
                    f"{name} must match joint_pos shape {tuple(joint_pos.shape)}, got {tuple(target_tensor.shape)}"
                )


@dataclass
class CameraState:
    """State of a single camera."""

    ## Images
    rgb: CameraRgbTensor | None
    """RGB image. Shape is (num_envs, H, W, 3)."""
    depth: CameraDepthTensor | None
    """Depth image. Shape is (num_envs, H, W)."""
    instance_id_seg: CameraSegmentationTensor | None = None
    """Instance id segmentation for each pixel. Shape is (num_envs, H, W)."""
    instance_id_seg_id2label: dict[int, str] | None = None
    """Instance id segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_id_seg`."""
    instance_seg: CameraSegmentationTensor | None = None
    """Instance segmentation for each pixel. Shape is (num_envs, H, W).

    .. warning::
        This is experimental and subject to change.
    """
    instance_seg_id2label: dict[int, str] | None = None
    """Instance segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_seg`.

    .. warning::
        This is experimental and subject to change.
    """

    ## Camera parameters
    pos: CameraPosTensor | None = None  # TODO: remove N
    """Position of the camera. Shape is (num_envs, 3)."""
    quat_world: CameraQuatTensor | None = None  # TODO: remove N
    """Quaternion ``(w, x, y, z)`` of the camera, following the world frame convention. Shape is (num_envs, 4).

    Note:
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.
    """
    intrinsics: CameraIntrinsicsTensor | None = None  # TODO: remove N
    """Intrinsics matrix of the camera. Shape is (num_envs, 3, 3)."""

    def __post_init__(self) -> None:
        num_envs = _maybe_num_envs_from_camera_state(self)

        if self.rgb is not None:
            rgb = _validate_rank("rgb", self.rgb, 4)
            _validate_last_dim("rgb", rgb, 3)
            num_envs = rgb.shape[0] if num_envs is None else num_envs
            _validate_num_envs("rgb", rgb, num_envs)

        if self.depth is not None:
            depth = _validate_rank("depth", self.depth, 3)
            num_envs = depth.shape[0] if num_envs is None else num_envs
            _validate_num_envs("depth", depth, num_envs)

        if self.instance_id_seg is not None:
            instance_id_seg = _validate_rank("instance_id_seg", self.instance_id_seg, 3)
            num_envs = instance_id_seg.shape[0] if num_envs is None else num_envs
            _validate_num_envs("instance_id_seg", instance_id_seg, num_envs)

        if self.instance_seg is not None:
            instance_seg = _validate_rank("instance_seg", self.instance_seg, 3)
            num_envs = instance_seg.shape[0] if num_envs is None else num_envs
            _validate_num_envs("instance_seg", instance_seg, num_envs)

        if self.pos is not None:
            pos = _validate_last_dim("pos", _validate_rank("pos", self.pos, 2), 3)
            num_envs = pos.shape[0] if num_envs is None else num_envs
            _validate_num_envs("pos", pos, num_envs)

        if self.quat_world is not None:
            quat_world = _validate_last_dim("quat_world", _validate_rank("quat_world", self.quat_world, 2), 4)
            num_envs = quat_world.shape[0] if num_envs is None else num_envs
            _validate_num_envs("quat_world", quat_world, num_envs)

        if self.intrinsics is not None:
            intrinsics = _validate_rank("intrinsics", self.intrinsics, 3)
            if intrinsics.shape[1:] != (3, 3):
                raise ValueError(f"intrinsics must have shape (num_envs, 3, 3), got {tuple(intrinsics.shape)}")
            num_envs = intrinsics.shape[0] if num_envs is None else num_envs
            _validate_num_envs("intrinsics", intrinsics, num_envs)

    @property
    def quat_ros(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the ROS convention. Shape is (num_envs, 4).

        Note:
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="ros")

    @property
    def quat_opengl(self) -> torch.Tensor:
        """Quaternion ``(w, x, y, z)`` of the camera, following the OpenGL convention. Shape is (num_envs, 4).

        Note:
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.
        """
        return convert_camera_frame_orientation_convention(self.quat_world, origin="world", target="opengl")


@dataclass
class TensorState:
    """Tensorized state of the simulation."""

    objects: dict[str, ObjectState]
    """States of all objects."""
    robots: dict[str, RobotState]
    """States of all robots."""
    cameras: dict[str, CameraState]
    """States of all cameras."""
    extras: dict = field(default_factory=dict)
    """States of Extra information"""

    def __post_init__(self) -> None:
        inferred_num_envs: int | None = None

        for state in self.objects.values():
            if inferred_num_envs is None:
                inferred_num_envs = state.root_state.shape[0]
            elif state.root_state.shape[0] != inferred_num_envs:
                raise ValueError(
                    f"TensorState objects must agree on num_envs={inferred_num_envs}, got {state.root_state.shape[0]}"
                )

        for state in self.robots.values():
            if inferred_num_envs is None:
                inferred_num_envs = state.root_state.shape[0]
            elif state.root_state.shape[0] != inferred_num_envs:
                raise ValueError(
                    f"TensorState robots must agree on num_envs={inferred_num_envs}, got {state.root_state.shape[0]}"
                )

        for state in self.cameras.values():
            camera_num_envs = _maybe_num_envs_from_camera_state(state)
            if camera_num_envs is None:
                continue
            if inferred_num_envs is None:
                inferred_num_envs = camera_num_envs
            elif camera_num_envs != inferred_num_envs:
                raise ValueError(
                    f"TensorState cameras must agree on num_envs={inferred_num_envs}, got {camera_num_envs}"
                )

        if isinstance(self.extras, dict) and inferred_num_envs is not None:
            for key, value in self.extras.items():
                if isinstance(value, torch.Tensor) and value.shape[0] != inferred_num_envs:
                    raise ValueError(
                        f"TensorState extras['{key}'] must have num_envs={inferred_num_envs}, got shape {tuple(value.shape)}"
                    )


StateOutput = Union[TensorState, DictStateBatch]


## Gymnasium types
Obs = Union[TensorState, torch.Tensor]
Reward = torch.Tensor
Success = torch.BoolTensor
TimeOut = torch.BoolTensor


class RawObservationInfo(TypedDict):
    obs: torch.Tensor


class ObservationInfo(TypedDict):
    raw: RawObservationInfo


class TaskInfo(TypedDict, total=False):
    privileged_observation: Obs
    episode_steps: torch.Tensor
    observations: ObservationInfo


InfoScalar = Union[bool, int, float, str, None]
InfoValue = Union[
    Obs,
    np.ndarray,
    InfoScalar,
    List["InfoValue"],
    Tuple["InfoValue", ...],
    Dict[str, "InfoValue"],
]
Info = Dict[str, InfoValue]
Termination = torch.BoolTensor
