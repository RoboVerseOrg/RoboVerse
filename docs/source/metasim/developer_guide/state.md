# State

MetaSim use `state` to describe the state of a simulation environment at a given time.
A unified state is the key to align different simulators.

## `state` Structure

The state is a dictionary that contains the following keys:
- `objects`: a dictionary that map object name to its state `object_state`.
- `robots`: a dictionary that map robot name to its state `robot_state`.
- `cameras`: a dictionary that map camera name to its state `camera_state`.

### `object_state` Structure

The `object_state` is a dictionary that contains the following keys:
- `pos`: the position of the object, as a `tensor([x, y, z])`.
- `rot`: the quaternion of the object, as a `tensor([w, x, y, z])`.
- `vel`: the linear velocity of the object, as a `tensor([vx, vy, vz])`.
- `ang_vel`: the angular velocity of the object, as a `tensor([wx, wy, wz])`.

The following keys are optional and only used for articulation objects:
- `dof_pos`: the joint positions, as a dict `{'joint1': qpos1, 'joint2': qpos2, ...}`.
- `dof_vel`: the joint velocities, as a dict `{'joint1': qvel1, 'joint2': qvel2, ...}`.
- `body`: a dictionary that maps body link name to its state `body_state`.

The `body_state` is a dictionary that contains `pos`, `rot`, `vel` and `ang_vel` keys. The definition is the same as above, but for the body link.


### `robot_state` Structure

The `robot_state` contains all the above keys of an articulation object. Plus, it also contains the following keys:
- `dof_pos_target`: the target joint positions, as a dict `{'joint1': qpos1, 'joint2': qpos2, ...}`.
- `dof_vel_target`: the target joint velocities, as a dict `{'joint1': qvel1, 'joint2': qvel2, ...}`.

### `camera_state` Structure

The `camera_state` is a dictionary that contains the following keys:
- `rgb`: the RGB images, as a tensor of shape `[H, W, 3]`.
- `depth`: the depth images, as a tensor of shape `[H, W]`.
- `pos`: the position of the camera, as a `tensor([x, y, z])`. (not supported yet)
- `look_at`: the look at point of the camera, as a `tensor([x, y, z])`. (not supported yet)
- `intrinsic`: the intrinsic matrix of the camera, as a tensor of shape `[3, 3]`. (not supported yet)
- `extrinsic`: the extrinsic matrix of the camera, as a tensor of shape `[4, 4]`. (not supported yet)

### State Example

Here is an feasible example of a state:

```python
{
    "objects": {
        "cube": {
            "pos": tensor([0.0, 0.0, 0.0]),
            "rot": tensor([1.0, 0.0, 0.0, 0.0]),
            "vel": tensor([0.0, 0.0, 0.0]),
            "ang_vel": tensor([0.0, 0.0, 0.0]),
        },
        "box": {
            "pos": tensor([0.0, 0.0, 0.0]),
            "rot": tensor([1.0, 0.0, 0.0, 0.0]),
            "vel": tensor([0.0, 0.0, 0.0]),
            "ang_vel": tensor([0.0, 0.0, 0.0]),
            "dof_pos": { "box_joint": 0.0 },
            "dof_vel": { "box_joint": 0.0 },
            "body": {
                "box_lid": {
                    "pos": tensor([0.0, 0.0, 0.0]),
                    "rot": tensor([1.0, 0.0, 0.0, 0.0]),
                    "vel": tensor([0.0, 0.0, 0.0]),
                    "ang_vel": tensor([0.0, 0.0, 0.0]),
                },
                "box_body": {
                    "pos": tensor([0.0, 0.0, 0.0]),
                    "rot": tensor([1.0, 0.0, 0.0, 0.0]),
                    "vel": tensor([0.0, 0.0, 0.0]),
                    "ang_vel": tensor([0.0, 0.0, 0.0]),
                },
            }
        },
    },
    "robots": {
        "franka": {
            "pos": tensor([0.0, 0.0, 0.0]),
            "rot": tensor([1.0, 0.0, 0.0, 0.0]),
            "vel": tensor([0.0, 0.0, 0.0]),
            "ang_vel": tensor([0.0, 0.0, 0.0]),
            "dof_pos": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785398,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.0,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            "dof_vel": {
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": 0.0,
                "panda_joint5": 0.0,
                "panda_joint6": 0.0,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
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
            },
            "dof_vel_target": {
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": 0.0,
                "panda_joint5": 0.0,
                "panda_joint6": 0.0,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
        }
    },
    "cameras": {
        "camera0": {
            "rgb": torch.zeros((H, W, 3)),
            "depth": torch.zeros((H, W)),
        }
    },
}
```

## `state` with Functions
MetaSim APIs always deal with `states` as a list of `state`. The length of the list is the number of environments. The observation term returned by `env.reset()` and `env.step()` is also unified to `states`.

- `handler.get_states() -> list[State]`
- `handler.set_states(states: list[State]) -> None`
- `env.reset(init_states: list[State]) -> tuple[list[State], Extra]`
- `env.step(actions: list[Action]) -> tuple[list[State], list[Reward], list[Success], list[TimeOut], Extra]`
