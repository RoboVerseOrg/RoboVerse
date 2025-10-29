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

