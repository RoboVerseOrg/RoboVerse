# Rerun Visualization Integration

This directory contains demos and utilities for visualizing RoboVerse simulations using [Rerun](https://rerun.io/).

Rerun is an open-source SDK for logging, storing, querying, and visualizing multimodal and multi-rate data. It provides a powerful viewer with timeline-based exploration, making it ideal for robotics simulation visualization.

## Features

- **URDF Robot Visualization**: Load and display robot models from URDF files
- **OBJ/Mesh Support**: Visualize mesh-based objects  
- **Primitive Shapes**: Cubes, spheres, cylinders with custom colors
- **Dynamic Updates**: Real-time pose and joint state updates during simulation
- **Timeline Playback**: Scrub through simulation history
- **Recording**: Save sessions as `.rrd` files for offline viewing
- **Camera Image Logging**: Display RGB and depth images from cameras

## Installation

Install the Rerun SDK:

```bash
pip install rerun-sdk
```

Optional dependencies for full functionality:
```bash
pip install yourdfpy trimesh
```

## Quick Start

### Replay Pre-recorded Task Demo (Recommended)

The easiest way to get started - replay existing task trajectories without needing GPU or IK solvers:

```bash
# Replay stack_cube task
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd

# Replay close_box task
python get_started/rerun/replay_task_demo.py --task close_box --sim mujoco --output close_box.rrd

# With live viewer during recording
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --spawn-viewer --output stack_cube.rrd

# View the saved recording
rerun stack_cube.rrd
```

### Basic Static Scene

```bash
# Using MuJoCo simulator
python get_started/rerun/rerun_demo.py --sim mujoco

# Using PyBullet simulator
python get_started/rerun/rerun_demo.py --sim pybullet
```

The Rerun viewer will open automatically and display the scene.

### Dynamic Simulation with IK

```bash
# Run dynamic simulation with robot motion
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic

# With PyRoKi IK solver (default)
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --solver pyroki

# With cuRobo IK solver
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --solver curobo
```

### Save Recording

```bash
# Save session to file for later replay
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --save-recording my_session.rrd

# Replay saved recording
rerun my_session.rrd
```

### CPU-Only Recording (No GPU Required)

For users on Mac or systems without GPU, use the simplified script that generates random joint motions directly (no IK solver needed):

```bash
# Sinusoidal joint motion (smooth, periodic)
python get_started/rerun/save_trajectory_simple.py --sim mujoco --output trajectory.rrd

# Random joint motion
python get_started/rerun/save_trajectory_simple.py --sim mujoco --motion-type random --output trajectory.rrd

# With Rerun viewer open during recording
python get_started/rerun/save_trajectory_simple.py --sim mujoco --spawn-viewer

# Replay saved recording
rerun trajectory.rrd
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | "franka" | Robot model to use |
| `--sim` | str | "mujoco" | Simulator backend (mujoco, pybullet, genesis, etc.) |
| `--num-envs` | int | 1 | Number of parallel environments |
| `--headless` | bool | True | Run simulator headless (use Rerun for visualization) |
| `--dynamic` | bool | False | Enable dynamic simulation with IK motion |
| `--solver` | str | "pyroki" | IK solver ("curobo" or "pyroki") |
| `--save-recording` | str | None | Path to save .rrd recording file |

## Viewer Controls

### Navigation
- **Rotate**: Left mouse drag
- **Pan**: Middle mouse drag or Shift+Left drag
- **Zoom**: Scroll wheel

### Timeline
- Use the timeline at the bottom to scrub through simulation history
- Play/pause button for automatic playback
- Adjust playback speed

### Entity Tree
- Toggle visibility of individual entities in the left panel
- Expand/collapse hierarchies to explore scene structure

## Integration with RL Training

Use the `TaskRerunWrapper` to add real-time visualization during RL training:

```python
from metasim.utils.rerun.rerun_env_wrapper import TaskRerunWrapper

# Wrap your environment
env = TaskRerunWrapper(
    task_env=your_env,
    app_name="RL Training Visualization",
    update_freq=10,  # Update every 10 steps
)

# Use normally
obs = env.reset()
for step in range(1000):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## Programmatic Usage

### Basic Visualization

```python
from metasim.utils.rerun.rerun_util import RerunVisualizer

# Initialize visualizer
visualizer = RerunVisualizer(
    app_name="My Robot Visualization",
    spawn=True,  # Auto-open viewer
)

# Add coordinate frame
visualizer.add_frame("world/origin")

# Visualize scenario items
visualizer.visualize_scenario_items(objects, object_states)
visualizer.visualize_scenario_items(robots, robot_states)
```

### Updating Poses

```python
# During simulation loop
for step in range(num_steps):
    # ... simulation code ...
    
    # Set time for timeline
    visualizer.set_time(step)
    
    # Update poses
    for name, state in robot_states.items():
        visualizer.update_item_pose(name, state)
```

### Logging Trajectories

```python
# Log individual points
visualizer.log_trajectory_point("ee_target", [0.5, 0.0, 0.6], color=[255, 0, 0])

# Log complete trajectory
positions = [[0.3, 0.0, 0.5], [0.4, 0.1, 0.5], [0.5, 0.2, 0.5]]
visualizer.log_trajectory("robot_path", positions, color=[0, 255, 0])
```

### Logging Camera Images

```python
# Log RGB and depth images
visualizer.log_camera_image("camera_1", rgb_image, depth_image)
```

## Comparison with Viser

| Feature | Rerun | Viser |
|---------|-------|-------|
| Viewer | Desktop app | Web browser |
| Timeline | Built-in scrubbing | Manual implementation |
| Recording | Native .rrd format | Custom |
| IK Control GUI | Not built-in | Built-in sliders |
| Joint Control GUI | Not built-in | Built-in sliders |
| Multi-modal data | Excellent | Limited |

**Choose Rerun when you need:**
- Timeline-based data exploration
- Recording and replay of sessions
- Multi-modal data logging (images, point clouds, etc.)
- Offline analysis

**Choose Viser when you need:**
- Interactive control via web GUI
- Real-time joint/IK manipulation
- Browser-based access

## Troubleshooting

### Viewer doesn't open
```bash
# Manually launch viewer
rerun

# Then connect from Python
from metasim.utils.rerun import RerunVisualizer
visualizer = RerunVisualizer(spawn=False, connect=True)
```

### URDF loading fails
- Ensure `yourdfpy` and `trimesh` are installed
- Check that mesh files are accessible from URDF paths

### Performance issues
- Reduce `update_freq` in `TaskRerunWrapper`
- Use `headless=True` to avoid duplicate rendering

## Files

- **`replay_task_demo.py`**: Replay pre-recorded task trajectories (CPU-only, no IK solver needed)
- **`rerun_demo.py`**: Main demo script with static and dynamic visualization
- **`save_trajectory.py`**: Save trajectory recording with IK solver (requires GPU)
- **`save_trajectory_simple.py`**: Save trajectory recording without IK solver (CPU-only, Mac compatible)
- **`../../metasim/utils/rerun/rerun_util.py`**: Core visualization utilities
- **`../../metasim/utils/rerun/rerun_env_wrapper.py`**: RL environment wrapper

## Additional Resources

- [Rerun Documentation](https://rerun.io/docs)
- [Rerun Python API](https://ref.rerun.io/docs/python/)
- [Rerun GitHub](https://github.com/rerun-io/rerun)
- [RoboVerse Viser Integration](../viser/README.md) - Alternative web-based visualizer

