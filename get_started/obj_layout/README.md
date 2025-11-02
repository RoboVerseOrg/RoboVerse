# Interactive Object Layout Control

This script provides an interactive keyboard control interface for real-time manipulation of objects and robots in the scene. **Press C to save all current poses to a file.**

## Quick Start

```bash
python get_started/obj_layout/object_layout_task.py
```

## Command Line Arguments

### Core Options
- `--task`: Task name (default: `put_banana`)
- `--robot`: Robot model (default: `franka`)
- `--sim`: Simulator backend (default: `isaacsim`)
  - Options: `isaacsim`, `genesis`, `mujoco`, `pybullet`, `sapien2`, `sapien3`, etc.

### Visualization
- `--enable-viser` / `--no-enable-viser`: Enable/disable Viser 3D web viewer (default: enabled)
  - Access at: `http://localhost:8080`
- `--display-camera` / `--no-display-camera`: Enable/disable local camera window (default: enabled)

### Physics
- `--enable-gravity`: Enable gravity for objects and robots (default: disabled)

## Usage Examples

```bash
# Basic usage (Viser + camera display enabled, gravity disabled)
python get_started/obj_layout/object_layout_task.py

# Minimal (no visualization)
python get_started/obj_layout/object_layout_task.py --no-enable-viser --no-display-camera

# With gravity
python get_started/obj_layout/object_layout_task.py --enable-gravity

# Different simulators
python get_started/obj_layout/object_layout_task.py --sim mujoco
python get_started/obj_layout/object_layout_task.py --sim genesis
```

## Keyboard Controls

### Main Controls
- **C**: 💾 **Save current poses** (one-key save!)
- **TAB**: Switch between objects/robots
- **J**: Toggle joint control mode
- **ESC**: Quit

### Position Control
- **↑/↓**: Move ±X
- **←/→**: Move ±Y
- **E/D**: Move ±Z (up/down)

### Rotation Control
- **Q/W**: Roll ±
- **A/S**: Pitch ±
- **Z/X**: Yaw ±

### Joint Control Mode (Press J)
- **↑/↓**: Increase/decrease angle
- **←/→**: Switch joint

## Output Files

**Press C** to save poses to `get_started/output/saved_poses_YYYYMMDD_HHMMSS.py`:

```python
poses = {
    "objects": {
        "banana": {
            "pos": torch.tensor([0.500000, 0.200000, 0.150000]),
            "rot": torch.tensor([0.000000, 0.000000, 0.000000, 1.000000]),
        },
    },
    "robots": {
        "franka": {
            "pos": torch.tensor([0.000000, 0.000000, 0.000000]),
            "rot": torch.tensor([0.000000, 0.000000, 0.000000, 1.000000]),
            "dof_pos": {"panda_joint1": 0.000000, ...},
        },
    },
}
```

## How to Use Saved Poses

The method depends on which script you're using:

### For `object_layout.py` (Handler-based Script)

This script uses **handler** directly. Apply saved poses by modifying the script:

```python
from saved_poses_20250129_143022 import poses

# In object_layout.py, add this line at the appropriate position
handler.set_states([poses] * num_envs)
```

The handler-based approach gives you direct control over state manipulation.

### For `object_layout_task.py` (Task-based Script)

This script uses a wrapped **Task**. You need to modify the task definition in `roboverse_pack/tasks/`:

1. **Find your task file**: `roboverse_pack/tasks/<category>/<task_name>.py`
   - Example: `roboverse_pack/tasks/embodiedgen/put_banana.py`

2. **Update the `scenario` class variable** to set initial positions and rotations:

```python
class PutBananaTask(Task):
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="banana",
                usd_path="...",
                pos=(0.5, 0.2, 0.15),  # ← From saved poses
                rot=(0, 0, 0, 1),      # ← From saved poses
            ),
        ],
        robots=[
            RobotCfg(
                name="franka",
                pos=(0.0, 0.0, 0.0),   # ← From saved poses
                rot=(0, 0, 0, 1),      # ← From saved poses
            ),
        ],
    )
```

3. **Update the `initial_state()` method** to set joint positions:

```python
    def initial_state(self) -> list[TensorState]:
        states = []
        for env_idx in range(self.num_envs):
            state = TensorState()
            # Set robot joint positions from saved poses
            state.robots["franka"].joint_pos = torch.tensor([0.0, -0.785, ...])
            states.append(state)
        return states
```

**Key Difference**: The handler-based script (`object_layout.py`) gives direct control, while the task-based script (`object_layout_task.py`) uses encapsulated Task classes for better organization and reusability.

## Requirements

- Python 3.8+
- PyTorch
- pygame
- OpenCV (optional, for `--display-camera`)
- Viser (optional, for `--enable-viser`)
- One of the supported simulators

## Notes

- Gravity is **disabled by default** for easier positioning
- Viser and camera display are **enabled by default**
- **Press C anytime** to save current layout
- Saved poses can be loaded in task configurations
