# 📦 Data Format v1 (Deprecated)

This document describes the **legacy v1 trajectory format** previously used in RoboVerse and related frameworks. This format is now deprecated in favor of the more extensible [Data Format v2](index.md#-1-collecting-trajectories-data-format-v2).

---

## 🔖 Typical Filename

```text
trajectory-unified.pkl
```

---

## 📄 Data Structure

```python
{
    'source': 'ManiSkill2-rigid_body-PickSingleYCB-v0',
    'max_episode_len': 147,
    'demos': {
        "franka": [
            {
                'name': 'banana_traj_0',
                'description': 'Banana nana bana ba banana.',  # optional
                'env_setup': {
                    "init_q": [...],          # [q_len], e.g., 9 for Franka
                    "init_robot_pos": [...],  # [3]
                    "init_robot_quat": [...], # [4]
                    "init_{obj_name}_pos": [...],    # [3]
                    "init_{obj_name}_quat": [...],   # [4]
                    "init_{joint_name}_q": [...],    # [1], for articulations
                    ...
                },
                'robot_traj': {
                    "q": [[...], ...],        # [demo_len x q_len]
                    "ee_act": [[...], ...],   # [demo_len x 1]  (0 = close, 1 = open)
                }
            },
            ...
        ],
    }
}
```

---

## 📌 Field Descriptions

| Key               | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `source`          | Environment string identifier                          |
| `max_episode_len` | Used to pad all trajectories for batching              |
| `demos`           | Dictionary mapping robot names to a list of demo dicts |
| `env_setup`       | Initial states of robot and objects in the environment |
| `q`    | Joint positions for each timestep                      |
| `ee_act`          | End-effector action per timestep (open/close gripper)  |

In each `env_setup`:

- `init_q`: Initial robot joint configuration
- `init_robot_pos`: Base robot position
- `init_robot_quat`: Base robot orientation (quaternion)
- `init_*`: Initial position/quaternion/state of relevant scene objects

These values are consumed inside the task's `_reset_idx` function.

---

## ⚠️ Status

This format is no longer supported in the latest RoboVerse infrastructure. If you have data in this format, use the provided script to convert it to v2:

```bash
python scripts/convert_traj_v1_to_v2.py --task CloseBox --robot franka
```

---


