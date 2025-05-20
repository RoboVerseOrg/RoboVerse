# Cross Embodiment

For parallel gripper on tabletop manipulation, you can specify different robot for the same task. For example, you can specify `iiwa` for `StackCube` task.

```bash
python metasim/scripts/replay_demo.py --sim=isaaclab --task=StackCube --num_envs=4 --robot=iiwa
```

## Retarget between Robots

We provide `src/scripts/retarget_demo.py` to retarget trajectories from one source robot to one or multiple target robots.

### Requirements

You need to go over:

- [Get Started / Installation / cuRobo Installation] for cuRobo
- Review [Data format v2](https://roboverse.wiki/metasim/developer_guide/new_task#data-format-v2) for the data format
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

The exported pickle file with contain the original demos as well as the retargetted demo for the target robots (see [Data format v2](https://roboverse.wiki/metasim/developer_guide/new_task#data-format-v2)):

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
