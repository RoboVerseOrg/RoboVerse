# Migrating New Tasks

This guide walks you through the full process of integrating a new task into RoboVerse, including trajectory preparation, asset conversion, configuration, and documentation.

---

## 📌 Overview

To add a new task, you need to complete the following four components:

1. **Trajectory** — Prepare a demonstration file in the v2 format  
2. **Assets** — Convert and organize USD assets  
3. **Configuration File** — Write a task config class in Python  
4. **Docstring** — Add structured documentation for your task

Each part is explained in detail below.

---

## 🔧 1. Collecting trajectories (Data Format v2)

Create a `.pkl` file containing demonstration data in **v2 format**. Make sure the filename ends with `_v2.pkl`. The data format is:

```python
{
    "franka": [
        {
            "actions": [
                {
                    "dof_pos_target": {"joint1": float, ...},
                    "ee_pose_target": {
                        "pos": [x, y, z],
                        "rot": [w, x, y, z],
                        "gripper_joint_pos": float
                    }
                },
                ...
            ],
            "init_state": {
                "object1": {"pos": [...], "rot": [...]},
                "robot": {"pos": [...], "rot": [...], "dof_pos": {...}},
                ...
            },
            "states": [state1, state2, ...],
            "extra": None
        },
        ...
    ],
    ...
}
```

Explanation:

- The relationship between actions and states:

```{mermaid}
graph LR
init_state --> a0["actions[0]"] --> s0["states[0]"] --> a1["actions[1]"] --> s1["states[1]"] --> ... --> an["actions[n-1]"] --> sn["states[n-1]"]
```

- `len(actions) == len(states)`
- Each object must be present in both `init_state` and every `states` entry.

### Convert v1 to v2

If your demonstrations were collected using the legacy **v1 format**, convert them to the new **v2 format** using:

```bash
python scripts/convert_traj_v1_to_v2.py --task CloseBox --robot franka
```

👉 For details about the v1 schema and field meanings, see [Data Format v1 (Deprecated)](./data_format_v1.md).

---

## 🧱 2. Preparing and Testing Assets

To define a new task in RoboVerse, you must prepare simulation assets in `USD`, `URDF`, or `MJCF` format. This section explains how to organize and validate them.

Assets should be placed in the following structure:

```
./roboverse_data/assets/<benchmark_name>/<task_name>/
```
To organize simulation assets for a task, use the following folder structure:

roboverse_data/
└── assets/
    └── <benchmark_name>/
        └── <task_name>/
            ├── obj1/
            │   ├── usd/
            │   │   └── obj1.usd
            │   ├── mjcf/
            │   │   └── obj1.mjcf
            │   ├── urdf/
            │   │   └── obj1.urdf
            │   └── textures/         # optional
            │       └── obj1_albedo.png
            ├── obj2/
            │   ├── usd/
            │   │   └── obj2.usd
            │   ├── mjcf/
            │   │   └── obj2.mjcf
            │   ├── urdf/
            │   │   └── obj2.urdf
            │   └── textures/         # optional
            │       └── obj2_albedo.jpg
            └── ...

---

### A. USD Assets 


#### Texture Guidelines

Ensure that all bitmap texture paths (e.g., Albedo Maps) in your USD files are **relative**:

```usd
diffuse_texture = "./textures/my_texture.png"  ✅
diffuse_texture = "/home/user/textures/my_texture.png"  ❌
```

📚 See [Omniverse Material Best Practices](https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html) for texture guidelines.

![materials](./images/material.jpg)

#### 🧪 Test USD Assets

You can validate your `.usd` asset by running:

```bash
python scripts/test_usd.py --usd_path {your_usd_file}
```

By default, this loads the asset as a rigid object.

> ✅ The test script must run **without errors**.\
> If your asset passes validation but fails to load in RoboVerse, please [open an issue](https://github.com/RoboVerseOrg/RoboVerse/issues).

---

### B. URDF Assets (Coming Soon)

---

### C. MJCF Assets (Coming Soon)

---

## ⚙️ 3. Write a Configuration File 

Create a new Python file under:

```
metasim/cfg/tasks/<benchmark_name>/<your_task>_cfg.py
```

It should define a task config class inheriting from `BaseTaskCfg`. Example:

```python
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg

class PickCubeCfg(BaseTaskCfg):
    task_name = "pick_cube"
    # Define scene elements, reward, success, randomization, etc.
```

### Register Your Task

After creating the config file, you must **register your task** by importing the config class in the `__init__.py` file under the corresponding benchmark directory:

For example, in:

```
metasim/cfg/tasks/rlbench/__init__.py
```

Add a line like:

```python
from .your_task_cfg import YourTaskCfg
```

> 📝 This ensures your task is included in the RoboVerse registry and can be loaded by name.\
> Without this, your task won't be discoverable or runnable.
---

## 📄 4. Add a Structured Docstring

Inside the task config file, write a docstring using the following format:

```python
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
```

| Field            | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| `title`          | Unique task name (must match registered name)                         |
| `group`          | Source dataset or benchmark (e.g. Maniskill, CALVIN)                  |
| `description`    | What this task does, and why it exists                                |
| `randomizations` | List of randomized elements in the environment                        |
| `success`        | Conditions for task completion and success                            |
| `badges`         | Tags like `demos`, `dense`, or `sparse`                               |
| `official_url`   | Link to original task documentation, if any                           |
| `poster_url`     | Optional path to image or gif showing the task                        |
| `video_url`      | Path to a short demo video (e.g. `pick_cube.mp4`)                     |
| `platforms`      | List of simulators supported by this task (e.g. `isaacgym`, `mujoco`) |
| `notes`          | Implementation differences or internal comments for other developers  |

ℹ️Note: This docstring format is required for correct indexing by RoboVerse. Improperly formatted docstrings will be ignored by the documentation system.

Also add your task to the documentation index:

```text
docs/source/metasim/api/metasim/metasim.cfg.tasks.rst
```

---

If all four components are correctly implemented, you can move on to verify the task using `replay_demo.py`.

---

## 🔗 References

The following resources may be useful when migrating existing datasets or assets into RoboVerse:

-  **CALVIN Trajectories**\
  GitHub: [https://github.com/Fisher-Wang/calvin](https://github.com/Fisher-Wang/calvin)\
  Provides tools and examples to convert CALVIN demonstration trajectories.

-  **RLBench Asset Export**\
  GitHub: [https://github.com/Fisher-Wang/RLBench\_export\_assets](https://github.com/Fisher-Wang/RLBench_export_assets)\
  Scripts for exporting RLBench assets (URDF / USD) in a format compatible with RoboVerse and Isaac Sim.

-  **RLBench Trajectories**\
  GitHub: [https://github.com/Fisher-Wang/RLBench](https://github.com/Fisher-Wang/RLBench)\
  Useful for converting RLBench demonstrations and behaviors into RoboVerse-compatible format.

-  [**Data Format v1 (Deprecated)**](./data_format_v1.md)\
  Full reference for the legacy v1 demonstration format and conversion guidance.

