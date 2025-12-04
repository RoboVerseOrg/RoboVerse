# 17. Rerun Visualization

This tutorial shows you how to use [Rerun](https://rerun.io/) to visualize RoboVerse simulations with timeline-based exploration, recording, and replay capabilities.

## What is Rerun?

Rerun is an open-source SDK for logging, storing, querying, and visualizing multimodal data. Unlike traditional simulation viewers, Rerun provides:

- **Timeline-based exploration**: Scrub through simulation history like a video
- **Recording & Replay**: Save sessions as `.rrd` files for offline viewing
- **Multi-modal support**: Visualize robots, objects, images, point clouds together
- **Cross-platform**: Works on Linux, macOS (including Apple Silicon), and Windows

## Installation

Install the Rerun SDK and dependencies:

```bash
pip install rerun-sdk trimesh yourdfpy
```

Verify installation:

```bash
rerun --version
```

## Quick Start

### Step 1: Replay an Existing Task Demo

The easiest way to get started is to replay a pre-recorded task trajectory. This doesn't require GPU or IK solvers.

```bash
# Replay the stack_cube task
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd
```

This will:
1. Load the `stack_cube` task configuration
2. Download required assets (URDF files, meshes)
3. Replay the trajectory step by step
4. Save the visualization as `stack_cube.rrd`

### Step 2: View the Recording

Open the saved recording in the Rerun viewer:

```bash
rerun stack_cube.rrd
```

You'll see:
- 🤖 **Franka robot** with moving joints
- 📦 **Colored cubes** being stacked
- ⏱️ **Timeline** at the bottom for scrubbing through the simulation

### Step 3: Live Viewer During Recording

To see the visualization in real-time while recording:

```bash
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd --spawn-viewer
```

## Available Demo Scripts

### 1. Replay Task Demo (Recommended for Beginners)

Replays pre-recorded task trajectories. **No GPU or IK solver needed!**

```bash
# Available tasks: stack_cube, close_box, pick_cube, etc.
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd

# Try different tasks
python get_started/rerun/replay_task_demo.py --task close_box --sim mujoco --output close_box.rrd
python get_started/rerun/replay_task_demo.py --task pick_cube --sim mujoco --output pick_cube.rrd
```

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | "stack_cube" | Task name to replay |
| `--robot` | str | "franka" | Robot to use |
| `--sim` | str | "mujoco" | Simulator backend |
| `--output` | str | "task_replay.rrd" | Output recording file |
| `--spawn-viewer` | bool | False | Open viewer during recording |
| `--max-steps` | int | None | Maximum steps to record |

### 2. Simple Trajectory Recording (CPU-Only)

Generates sinusoidal or random joint motions directly. **Works on Mac without GPU!**

```bash
# Sinusoidal motion (smooth, periodic)
python get_started/rerun/save_trajectory_simple.py --sim mujoco --output trajectory.rrd

# Random motion
python get_started/rerun/save_trajectory_simple.py --sim mujoco --motion-type random --output trajectory.rrd

# More simulation steps
python get_started/rerun/save_trajectory_simple.py --sim mujoco --num-steps 500 --output trajectory.rrd
```

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | "franka" | Robot model |
| `--sim` | str | "mujoco" | Simulator backend |
| `--output` | str | "trajectory.rrd" | Output recording file |
| `--num-steps` | int | 200 | Number of simulation steps |
| `--motion-type` | str | "sinusoidal" | Motion type: "sinusoidal" or "random" |
| `--spawn-viewer` | bool | False | Open viewer during recording |

### 3. Full Demo with IK Solver (Requires GPU)

For users with GPU and IK solver (PyRoKi or cuRobo):

```bash
# With PyRoKi IK solver
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --solver pyroki

# With cuRobo IK solver
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --solver curobo

# Save recording
python get_started/rerun/rerun_demo.py --sim mujoco --dynamic --save-recording demo.rrd
```

## Step-by-Step Tutorial

### Understanding the Rerun Viewer

When you open a `.rrd` file, the Rerun viewer shows:

```
┌─────────────────────────────────────────────────────────────┐
│  [Entity Tree]  │           [3D Viewport]                   │
│                 │                                           │
│  ▼ world        │     🤖 Robot + 📦 Objects                 │
│    ▼ franka     │                                           │
│      panda_link0│                                           │
│      panda_link1│                                           │
│      ...        │                                           │
│    ▼ cube       │                                           │
│    ▼ base       │                                           │
│                 │                                           │
├─────────────────┴───────────────────────────────────────────┤
│  [Timeline]  ◀ ▶ ━━━━━●━━━━━━━━━━━━━━━━━━  Step: 42/200     │
└─────────────────────────────────────────────────────────────┘
```

**Navigation Controls:**
- **Rotate**: Left mouse drag
- **Pan**: Middle mouse drag or Shift+Left drag  
- **Zoom**: Scroll wheel
- **Timeline**: Drag the playhead or use play button

### Creating Your Own Visualization

Here's how to add Rerun visualization to your own simulation:

```python
from metasim.utils.rerun.rerun_util import RerunVisualizer

# 1. Initialize visualizer
visualizer = RerunVisualizer(
    app_name="My Simulation",
    spawn=True,           # Auto-open viewer
    save_path="my_sim.rrd"  # Optional: save recording
)

# 2. Add coordinate frame (optional)
visualizer.add_frame("world/origin")

# 3. Initial visualization of objects and robots
visualizer.visualize_scenario_items(scenario.objects, object_states)
visualizer.visualize_scenario_items(scenario.robots, robot_states)

# 4. Simulation loop
for step in range(num_steps):
    # Set timeline position
    visualizer.set_time(step)
    
    # Run simulation
    handler.simulate()
    obs = handler.get_states(mode="tensor")
    
    # Extract and update states
    for name, state in robot_states.items():
        visualizer.update_item_pose(name, state)
    for name, state in object_states.items():
        visualizer.update_item_pose(name, state)

# 5. Cleanup
visualizer.close()
```

### State Format

The state dictionary format expected by `update_item_pose`:

```python
state = {
    "pos": [x, y, z],           # Position in world frame
    "rot": [w, x, y, z],        # Quaternion (wxyz format)
    "dof_pos": {                # Joint positions (for articulated objects)
        "joint_name": value,
        ...
    }
}
```

## Comparison with Other Visualizers

| Feature | Rerun | Native Viewer | Viser |
|---------|-------|---------------|-------|
| Timeline scrubbing | ✅ | ❌ | ❌ |
| Recording/Replay | ✅ `.rrd` | ❌ | ❌ |
| Works on Mac | ✅ | ⚠️ Limited | ✅ |
| No GPU required | ✅ | ⚠️ | ✅ |
| Interactive controls | ❌ | ✅ | ✅ |
| Web-based | ❌ | ❌ | ✅ |

**Use Rerun when:**
- You need to record and replay simulations
- You want timeline-based exploration
- You're debugging complex trajectories
- You're on macOS without GPU

**Use Native Viewer when:**
- You need real-time interactive simulation
- You want built-in physics visualization

**Use Viser when:**
- You need web-based access
- You want interactive joint/IK sliders

## Troubleshooting

### Viewer doesn't open automatically

```bash
# Launch viewer manually
rerun

# Then run your script with connect mode
python your_script.py
```

### URDF/mesh loading issues

Ensure dependencies are installed:
```bash
pip install trimesh yourdfpy
```

### Recording file too large

Reduce the number of steps or recording frequency:
```bash
python get_started/rerun/replay_task_demo.py --task stack_cube --max-steps 100 --output small.rrd
```

### Performance issues on macOS

Use headless mode for the simulator:
```bash
python get_started/rerun/replay_task_demo.py --task stack_cube --sim mujoco --output stack_cube.rrd
# The simulator runs headless, only Rerun viewer shows
```

## Example Output

After running the stack_cube demo, you'll see:

| Rerun Viewer |
|:---:|
| ![Rerun Viewer](../../../_static/standard_output/rerun_stack_cube.png) |

The recording shows:
- Franka robot arm with all links and joints
- Red cube being picked up
- Blue base cube as the target
- Full timeline for scrubbing through the trajectory

## Files Reference

| File | Description |
|------|-------------|
| `get_started/rerun/replay_task_demo.py` | Replay pre-recorded task trajectories |
| `get_started/rerun/save_trajectory_simple.py` | CPU-only trajectory recording |
| `get_started/rerun/rerun_demo.py` | Full demo with IK solver |
| `metasim/utils/rerun/rerun_util.py` | Core RerunVisualizer class |
| `metasim/utils/rerun/rerun_env_wrapper.py` | RL environment wrapper |

## Next Steps

- Try different tasks: `close_box`, `pick_cube`, `poke_cube`
- Add Rerun visualization to your own training loop
- Explore the [Rerun documentation](https://rerun.io/docs) for advanced features
- Check out the [Viser integration](../advanced/viser/usage.md) for web-based visualization

