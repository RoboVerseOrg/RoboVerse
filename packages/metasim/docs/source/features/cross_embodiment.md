# Cross-Embodiment Transfer

Cross-embodiment enables policies and demonstrations to be transferred across different robot morphologies.

See the [RoboVerse robots dataset](https://roboverse.wiki/roboverse/dataset_benchmark/dataset/robots.html) for the full list of supported robots.

---

## Basic Usage

### Switching Robots for Tasks

For tabletop manipulation tasks with parallel grippers, you can swap robots easily:

```bash
# Run StackCube with Franka (default)
python scripts/advanced/replay_demo.py --sim=isaaclab --task=StackCube --num_envs=4

# Run the same task with KUKA iiwa
python scripts/advanced/replay_demo.py --sim=isaaclab --task=StackCube --num_envs=4 --robot=iiwa

# Run with UR10
python scripts/advanced/replay_demo.py --sim=isaaclab --task=StackCube --num_envs=4 --robot=ur10
```

---

## Retarget between Robots

Trajectory retargeting between robots was provided by `scripts/advanced/retarget_demo.py`, which targeted a pre-1.0 API and has been removed; a maintained replacement is tracked in the roadmap (`TODO: verify` before relying on the command below, which documents the former interface).

### Requirements

You need to go over:

- [Get Started / Installation / cuRobo Installation] for cuRobo
- Review [Trajectory](../concept/state.md#trajectory) for the v2 trajectory format
- Make sure that the following items are carefully set in the robots' meta configs:
  - `gripper_open_q` / `gripper_close_q`: A list specifying the gripper's joint positions when it releases / grasps the object
  - `curobo_ref_cfg_name`: cuRobo config file for the robot
  - `curobo_tcp_rel_pos` / `curobo_tcp_rel_rot`: Relative transformation from the TCP frame to the EE frame
    - The "EE frame" here is the `ee_link` specified by the cuRobo config

```python
@configclass
class BaseRobotMetaCfg(ArticulationObjMetaCfg):
    # ...

    gripper_open_q: list[float] = MISSING
    gripper_close_q: list[float] = MISSING

    # cuRobo Configs
    curobo_ref_cfg_name: str = MISSING
    curobo_tcp_rel_pos: tuple[float, float, float] = MISSING
    curobo_tcp_rel_rot: tuple[float, float, float] = MISSING
```


### Source Data and Configurations Preparation

To perform cross-embodiment retarget, you need to get robot configurations for the source and all the target robots prepared. You also need a demo data (`.pkl`) that contains the trajectory.

The robot meta config should include the information about the Tool Center Point (TCP) frame: On which link's frame is it defined, and the relative transformation. Ideally, if the TCP link is already defined, you can

### Retarget

```shell
python src/scripts/retarget_demo.py --source_path data_isaaclab/source_data/maniskill2/rigid_body/PickCube-v0/trajectory-unified_v2.pkl --source_robot franka --target_robots iiwa franka_with_gripper_extension
```

The exported pickle file contains the original demos as well as the retargeted demo for the target robots. See [Trajectory](../concept/state.md#trajectory) for the v2 trajectory format:

```
{
    "franka": [  // robot name should be same as BaseRobotMetaCfg.name
        // Demo for Franka
        "actions": [ ... ],
        "init_state": { ... },
        "states": [ ... ]
        "extra": ...
	],
	"iiwa": [
       ... // Demo for KUKA IIWA
	],
	"ur10": [
       ... // Demo for UR10
	]
}
```

By specifying `--viz`, the first retargetted trajectory will be visualized via plotly in your browser.
