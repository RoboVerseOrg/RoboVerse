# States, Actions and Trajectories

## Overview

States are how we communicate with simulated environments. We provide two state types: `TensorState` and `DictEnvState` . `TensorState` is implemented with torch tensors as base data structure and designed for efficiency, while `DictEnvState` is implemented with dicts for more user-friendly interactions.

Actions are how we control robots in the simulation. The most basic action type is joint angle control, where you directly specify target joint positions for the robot.

When multiple states are organized as a sequence, we will have a `Trajectory`.

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

## Actions

Actions are used to control robots in the simulation. The most fundamental action type is **joint angle control**, where you directly specify target joint positions for each joint.

### Basic Joint Angle Control

Actions are passed to the simulator as a **list of dictionaries**, where each dictionary corresponds to one environment. The basic format for joint angle control is:

```python
actions = [
    {
        <robot_name>: {
            "dof_pos_target": {
                <joint_name_1>: <target_value_1>,
                <joint_name_2>: <target_value_2>,
                ...
            }
        }
    },
    # ... more dicts for additional environments
]
```

You can apply actions using `handler.set_dof_target(actions)`.

### Example: Franka Panda Robot Control

Here's a concrete example of controlling a Franka Panda robot. The Franka has 7 arm joints and 2 finger joints:

```python
# Example 1: Set Franka to home position, num_envs=1
actions = [{
    "franka": {
        "dof_pos_target": {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785398,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356194,
            "panda_joint5": 0.0,
            "panda_joint6": 1.570796,
            "panda_joint7": 0.785398,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
    }
}]
handler.set_dof_target(actions)
```

```python
# Example 2: Random joint positions within joint limits, num_envs=1
import torch

actions = [{
    "franka": {
        "dof_pos_target": {
            joint_name: (
                torch.rand(1).item()
                * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                + robot.joint_limits[joint_name][0]
            )
            for joint_name in robot.joint_limits.keys()
        }
    }
}]
handler.set_dof_target(actions)
```

```python
# Example 3: Multiple environments with different actions, num_envs=3
actions = [
    {
        "franka": {
            "dof_pos_target": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785398,
                # ... other joints
            }
        }
    },
    {
        "franka": {
            "dof_pos_target": {
                "panda_joint1": 0.5,
                "panda_joint2": -1.0,
                # ... other joints
            }
        }
    },
    {
        "franka": {
            "dof_pos_target": {
                "panda_joint1": -0.5,
                "panda_joint2": -0.5,
                # ... other joints
            }
        }
    },
]
handler.set_dof_target(actions)
```

### Important Notes

- **List of Dictionaries**: Actions must be provided as a list of dictionaries, where the length of the list equals `num_envs`. Each dictionary specifies the actions for one environment.
- **No Scaling by Default**: The action values are directly applied as target joint positions. There is no normalization or scaling by default (e.g., no automatic mapping from [-1, 1] to joint limits).
- **Joint Limits**: Make sure your target values are within the robot's joint limits. You can access joint limits via `robot.joint_limits`.
- **Partial Updates**: You can specify only a subset of joints in the action. Unspecified joints will maintain their previous target values.

### Tensor Format (Advanced)

For efficiency in vectorized operations, `set_dof_target` also supports a **2D torch tensor** format with shape `(num_envs, num_actuators)` instead of list of dictionaries:

```python
import torch

# Shape: (num_envs, num_actuators)
# Joint values are ordered by joint name in alphabetical (dictionary) order
# For Franka: [panda_finger_joint1, panda_finger_joint2, panda_joint1, ..., panda_joint7]
actions_tensor = torch.tensor([
    [0.04, 0.04, 0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],  # env 0
    [0.04, 0.04, 0.5, -1.0, 0.2, -2.0, 0.1, 1.5, 0.8],  # env 1
    # ... more environments
])

handler.set_dof_target(actions_tensor)
```

**Important**: When using tensor format, the joint values must be ordered by **joint name in alphabetical (dictionary) order**. For Franka Panda, this order is:
1. `panda_finger_joint1`
2. `panda_finger_joint2`
3. `panda_joint1`
4. `panda_joint2`
5. `panda_joint3`
6. `panda_joint4`
7. `panda_joint5`
8. `panda_joint6`
9. `panda_joint7`

You can check the exact order using `sorted(robot.joint_limits.keys())`.

### Higher-Level Control

For higher-level control modes such as:
- End-effector (EE) position/orientation control
- Delta joint control (incremental movements)
- Velocity control
- Torque control

These can be implemented in your task's `step()` function by converting the high-level actions to joint-level commands. The base simulator interface uses direct joint angle control as the fundamental action space.

## Trajectory

A `Trajectory` is a file that contains a sequence of states.

In Metasim, we have different versions of Trajectory formats. The most recent one is the v2 trajectory format, which is a dumped dict:

```python
trajs = {
    <robot_name>: [
        DictTraj1,
        DictTraj2,
        ...
    ]
}
```

Where `DictTraj` is a dict with the following structure:

```python
dict_traj = {
    "actions": List[Dict[str, np.ndarray]],  # List of actions
    "states": [
        DictTrajState1,
        DictTrajState2,
        ...
    ]
}
```

Where `DictTrajState` is a dict with the following structure:

```python
dict_traj_state = {
    <object_name1>: {
        "pos": np.ndarray,
        "rot": np.ndarray,
    },
    <object_name2>: {
        "pos": np.ndarray,
        "rot": np.ndarray,
    },
    ...
}
```

Trajectories can be loaded with `metasim.utils.demo_util.loader.load_traj_file()`, and can be saved with `metasim.utils.demo_util.loader.save_traj_file()`.