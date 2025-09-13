# States

## Overview

States are how we communicate with simulated environments. We provide two state types: `TensorState` and `DictEnvState` . `TensorState` is implemented with torch tensors as base data structure and designed for efficiency, while `DictEnvState` is implemented with dicts for more user-friendly interactions.

## Tensor State

`TensorStates` is a dict of vectorized states. At the top level, it contains 4 fields: objects, robots, cameras and extras. Each field then contains multiple sub-fields for data.

```python
@dataclass
class ObjectState:
    """State of a single object."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str] | None = None
    """Body names. This is only available for articulation objects."""
    body_state: torch.Tensor | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
    joint_pos: torch.Tensor | None = None
    """Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
    joint_vel: torch.Tensor | None = None
    """Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""

  @dataclass
class RobotState:
    """State of a single robot."""

    root_state: torch.Tensor
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str]
    """Body names."""
    body_state: torch.Tensor
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: torch.Tensor
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: torch.Tensor
    """Joint velocities. Shape is (num_envs, num_joints)."""
    joint_pos_target: torch.Tensor
    """Joint positions target. Shape is (num_envs, num_joints)."""
    joint_vel_target: torch.Tensor
    """Joint velocities target. Shape is (num_envs, num_joints)."""
    joint_effort_target: torch.Tensor
    """Joint effort targets. Shape is (num_envs, num_joints)."""

@dataclass
class CameraState:
    """State of a single camera."""

    ## Images
    rgb: torch.Tensor | None
    """RGB image. Shape is (num_envs, H, W, 3)."""
    depth: torch.Tensor | None
    """Depth image. Shape is (num_envs, H, W)."""
    instance_id_seg: torch.Tensor | None = None
    """Instance id segmentation for each pixel. Shape is (num_envs, H, W)."""
    instance_id_seg_id2label: dict[int, str] | None = None
    """Instance id segmentation id to label mapping. Keys are instance ids, values are labels. Go together with :attr:`instance_id_seg`."""
    instance_seg: torch.Tensor | None = None
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
    pos: torch.Tensor | None = None  # TODO: remove N
    """Position of the camera. Shape is (num_envs, 3)."""
    quat_world: torch.Tensor | None = None  # TODO: remove N
    """Quaternion ``(w, x, y, z)`` of the camera, following the world frame convention. Shape is (num_envs, 4).

    Note:
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.
    """
    intrinsics: torch.Tensor | None = None  # TODO: remove N
    """Intrinsics matrix of the camera. Shape is (num_envs, 3, 3)."""

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
```

To obtain the `TensorState` from a handler, one can use `handler.get_state(mode="tensor")` method. The return value will be a `TensorState` describing the current simulation status.

Then, you can access the `TensorState` with:

```python
tensor_state = handler.get_state(mode="tensor")
object_pos = tensor_state.objects["ball"].root_state[:, 0:3]  # root_state.shape = (num_envs, 13)
```

`TensorState` instances can also be passed into the handler to set the simulation state.

Metasim `TaskWrappers` by default encourages the user to write the `observation()` function to return `TensorState` too.

One disadvantage of `TensorState` is that it is diffucult for human users to undestand the mapping between tensor indices and actual states. So we also provide a more user-friendly interface.

## Dict State

`DictState` is more user-friendly compared to `TensorState`, but scrifices efficiency.

```python
Dof = Dict[str, float]

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
    extras: dict[str, Any]      # States of Extra information
```

To obtain the `DictEnvState` from a handler, one can use `handler.get_state(mode="dict")` method. The return value will be a `DictEnvState` describing the current simulation status.

Then, you can access the `DictEnvState` with:

```python
dict_state = handler.get_state(mode="dict")
object_pos = dict_state["objects"]["ball"]["pos"]
```

`DictEnvState` instances can also be passed into the handler to set the simulation state.

The disadvantage of `DictState` is its speed. In some cases, `DictState` is extremely slow due to frequent dict access and discontinuous memory access.
