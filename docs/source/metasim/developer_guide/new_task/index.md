# Migrating New Tasks

This guide walks you through the full process of integrating a new task into RoboVerse, including trajectory preparation, asset conversion, configuration, and documentation.

---

## 📌 Overview

To add a new task, you need to complete the following four components:

1. **Trajectory (`traj`)** — Prepare a demonstration file in the v2 format  
2. **Assets** — Convert and organize USD assets  
3. **Configuration File** — Write a task config class in Python  
4. **Docstring** — Add structured documentation for your task

Each part is explained in detail below.

---

## 🔧 1. Collecting trajectories (Data Format v2)


Create a `.pkl` file containing demonstration data in **v2 format**. If the filename ends with `_v2.pkl`, the demo reader will automatically parse it using the v2 schema. The data format is:

```
{
    "franka": [  // robot name should be same as BaseRobotMetaCfg.name
        {
            "actions": [
                {
                    // one or more of the following
                    "dof_pos_target": {
                        "{joint_name1}": float,
                        "{joint_name2}": float,
                        ...
                    },
                    "ee_pose_target": {
                        "pos": [float, float, float],
                        "rot": [float, float, float, float],  // (w, x, y, z)
                        "gripper_joint_pos": float,
                    }
                },
                ...
            ],
            "init_state": {
                "{obj_name1}": {  // example rigid object
                    "pos": [float, float, float],
                    "rot": [float, float, float, float],  // (w, x, y, z)
                },
                "{obj_name2}": {  // example articulation object
                    "pos": [float, float, float],
                    "rot": [float, float, float, float],  // (w, x, y, z)
                    "dof_pos": {
                        "{joint_name1}": float,
                        "{joint_name2}": float,
                        ...
                    }
                },
                "{robot_name}": {  // robot name should be same as BaseRobotMetaCfg.name
                    "pos": [float, float, float],
                    "rot": [float, float, float, float],  // (w, x, y, z)
                    "dof_pos": {
                        "{joint_name1}": float,
                        "{joint_name2}": float,
                        ...
                    }
                },
                ...
            },
            "states": [state1, state2, ...]  // list of states, a state has the same format as the init_state
            "extra": None  // extra information for specific use, default is None
        },
        ...
    ],
    ...
}
```
Explaination:
- The relationship between actions and states:
    ```{mermaid}
    graph LR
    init_state --> a0["actions[0]"] --> s0["states[0]"] --> a1["actions[1]"] --> s1["states[1]"] --> ... --> an["actions[n-1]"] --> sn["states[n-1]"]
    ```
- `len(actions) == len(states)`
- Every object should have a key in the `init_state` and `states` dict.

### Convert v1 to v2
If you have already exported the trajectory data in v1 format, you can convert it to v2 format by:
```bash
python scripts/convert_traj_v1_to_v2.py --task CloseBox --robot franka
```

---
## 🧱 2. Preparing and Testing Assets
To define a new task, you must prepare the simulation assets in `.usd` format and organize them in the following directory:

```
./data_isaaclab/assets/<benchmark_name>/<task_name>/
```

### 🔄 Converting Assets to USD

RoboVerse relies on USD assets. If your original files are in URDF, MJCF, or mesh formats, use the provided script to convert them:

```bash
python scripts/convert_usd.py --input {your_file}
```

> 📝 This ensures compatibility with the Isaac Lab standard.  
> If you're migrating from an older format, this step is required.

You can also refer to the official [Isaac Lab Asset Import Guide](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) for more details.

### 🎨 Texture Paths

Ensure all bitmap texture paths (e.g., Albedo Maps) are **relative paths**, not absolute.  
For example:

```usd
diffuse_texture = "./textures/my_texture.png"  ✅
diffuse_texture = "/home/user/textures/my_texture.png"  ❌
```

📚 See [Omniverse Material Best Practices](https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html) for texture guidelines.

![materials](./images/material.jpg)

### 🧪 Test Assets

You can validate your `.usd` asset by running:

```bash
python scripts/test_usd.py --usd_path {your_usd_file}
```

By default, the asset is loaded as a rigid object.

For more options, run:

```bash
python scripts/test_usd.py --help
```

> ✅ The test script must run **without errors**.  
> If your asset passes validation but still fails in RoboVerse, please [open an issue](https://github.com/RoboVerseOrg/RoboVerse/issues).

## ⚙️ 3. Write a Configuration File (`cfg`)

Create a new Python file under:

```
metasim/cfg/tasks/<your_group>/<your_task>_cfg.py
```

It should define a task config class inheriting from `BaseTaskCfg`. Example:

```python
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg

class PickCubeCfg(BaseTaskCfg):
    task_name = "pick_cube"
    # Define scene elements, reward, success, randomization, etc.
```

Make sure your task is properly registered in the task registry.

---

## 📄 4. Add a Structured Docstring

Inside the task config file, write a docstring using the following format:

````python
"""Pick up a red cube and move it to the goal.

.. Description::

### title:
pick_cube

### group:
maniskill

### description:
A simple pick-and-place task with a red cube and a fixed goal.

### randomizations:
- Cube XY position
- Goal Z height

### success:
- Cube within 2.5cm of goal
- Robot velocity < 0.2

### badges:
- demos
- sparse

### video_url:
pick_cube.mp4

### platforms:
isaaclab, mujoco

### notes:
Imported from ManiSkill and adapted to IsaacLab format.
"""
````

Also add your task to the documentation index:

```text
docs/source/metasim/api/metasim/metasim.cfg.tasks.rst
```

---

If all four components are correctly implemented, you can move on to verify the task using `replay_demo.py`.