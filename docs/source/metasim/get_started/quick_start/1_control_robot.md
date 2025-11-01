#  1. Control Robot
In this tutorial, we will show you how to use MetaSim to control a robot.

## Common Usage

```bash
python get_started/1_control_robot.py  --sim <simulator>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will give random control actions to the robot and it will automatically record a video.


### Examples

#### IsaacSim
```bash
python get_started/1_control_robot.py  --sim isaacsim
```

#### Isaac Gym
```bash
python get_started/1_control_robot.py  --sim isaacgym
```

#### Mujoco
```bash
# For mac users, replace python with mjpython.
python get_started/1_control_robot.py  --sim mujoco --headless
```
Note that we find the `non-headless` mode of Mujoco is not stable. So we recommend using the `headless` mode.

#### Genesis
```bash
python get_started/1_control_robot.py  --sim genesis
```
Note that we find the `headless` mode of Genesis is not stable. So we recommend using the `non-headless` mode.

#### Sapien
```bash
python get_started/1_control_robot.py  --sim sapien3
```

#### Pybullet
```bash
python get_started/1_control_robot.py  --sim pybullet
```


You will get the following videos:

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">IsaacSim</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_mujoco.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">MuJoCo</p>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; width: 100%;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_genesis.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Genesis</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_sapien3.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">SAPIEN</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/1_move_robot_pybullet.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">PyBullet</p>
        </div>
    </div>
</div>

## Code Highlights

**Robot Control**: Use `handler.set_dof_targets(actions)` to control robot joints. Actions follow the format:
```python
actions = [
    {
        robot.name: {
            "dof_pos_target": {
                joint_name: (
                    torch.rand(1).item()
                    * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                    + robot.joint_limits[joint_name][0]
                )
                for joint_name in robot.joint_limits.keys()
            }
        }
    }
    for _ in range(scenario.num_envs)
]
handler.set_dof_targets(actions)
handler.simulate()
```

**Key Points**:
- Can also use direct tensor actions instead of dictionary format ，Action order matches `robot.actuators` and `robot.joint_limits` configuration
- `set_dof_targets()` only sets targets - call `handler.simulate()` to step physics
- Unlike Task's `step()`, Handler requires separate `simulate()` call
