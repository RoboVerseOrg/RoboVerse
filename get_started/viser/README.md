# Viser Visualization Demo

This directory contains a unified demo for viser-based 3D visualization and robot control.

## Files

- **`viser_demo.py`**: Unified demo script with all features
- **`viser_util.py`**: Core visualization utilities and ViserVisualizer class
- **`VISER_USAGE_GUIDE.md`**: Comprehensive usage guide

## Quick Start

### Prerequisites
```bash
cd /home/xy/Documents/RoboVerse
conda activate isaacsim  # or your environment
```

### Basic Static Scene
```bash
python get_started/viser/viser_demo.py --sim pybullet
```
Open http://localhost:8080 to view the scene.

## Features

### 1. Static/Dynamic Scene Visualization
- **Static**: Display scene without simulation
- **Dynamic**: Run IK-based robot motion with simulation

### 2. Interactive Joint Control
Control robot joints with GUI sliders:
```bash
python get_started/viser/viser_demo.py --sim pybullet --enable-joint-control
```

### 3. Interactive IK Control
Control end-effector position/orientation:
```bash
python get_started/viser/viser_demo.py --sim pybullet --enable-ik-control
```

### 4. Trajectory Playback
Load and play recorded trajectories:
```bash
python get_started/viser/viser_demo.py --sim pybullet --enable-trajectory \
    --trajectory-path roboverse_data/trajs/rlbench/close_box/v2/franka_v2.pkl.gz
```

### 5. Dynamic Simulation
Run IK-based robot motion:
```bash
python get_started/viser/viser_demo.py --sim pybullet --dynamic
```

With video recording:
```bash
python get_started/viser/viser_demo.py --sim pybullet --dynamic --save-video
```

### 6. Combine Multiple Features
```bash
python get_started/viser/viser_demo.py --sim pybullet \
    --enable-joint-control \
    --enable-ik-control \
    --enable-trajectory \
    --trajectory-path roboverse_data/trajs/rlbench/close_box/v2/franka_v2.pkl.gz
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | "franka" | Robot model to use |
| `--sim` | str | "mujoco" | Simulator backend (mujoco, pybullet, etc.) |
| `--num-envs` | int | 1 | Number of parallel environments |
| `--headless` | bool | True | Run simulator headless |
| `--dynamic` | bool | False | Enable dynamic simulation with IK motion |
| `--enable-joint-control` | bool | False | Enable interactive joint control |
| `--enable-ik-control` | bool | False | Enable interactive IK control |
| `--enable-trajectory` | bool | False | Enable trajectory playback |
| `--trajectory-path` | str | None | Path to trajectory file (.pkl.gz) |
| `--solver` | str | "pyroki" | IK solver ("curobo" or "pyroki") |
| `--save-video` | bool | False | Save video of dynamic simulation |

## Key Features

- **Unified Demo** - Single script replaces 5+ separate demos  
- **Flexible Configuration** - Enable/disable features via CLI flags  
- **No Code Duplication** - Shared scene setup and utilities  
- **Static + Dynamic** - Both modes in one script  
- **Multiple Control Modes** - Joint, IK, Trajectory controls  
- **Reset to Initial Pose** - All control modes support resetting to demo-defined initial pose

## GUI Controls

### Camera Controls
- **Rotate**: Left mouse drag
- **Pan**: Right mouse drag or Shift+Left drag  
- **Zoom**: Scroll wheel
- **Preset Views**: Top, Side, Front buttons
- **Screenshot**: Capture button
- **Video Recording**: Start/Stop recording

### Joint Control
1. Select robot from dropdown
2. Click "Setup Joint Control"
3. Use individual joint sliders
4. Click "Reset Joints" to return to initial pose
5. Click "Clear Joint Control" to remove GUI

### IK Control
1. Click "Setup IK Control"
2. Adjust target position sliders (X, Y, Z)
3. Adjust target orientation sliders (Quat W, X, Y, Z)
4. Visual markers show target (red sphere + RGB axes)
5. Click "Solve & Apply IK" to move robot
6. Click "Reset Robot Joints" to return to initial pose
7. Click "Reset Target" to reset target markers

### Trajectory Playback
1. Load trajectory via `--trajectory-path` argument
2. Click "Update Robot List" to refresh
3. Select robot and demo index
4. Click "Set Current Trajectory"
5. Use Play/Pause/Stop controls
6. Drag timeline slider to seek
7. Adjust "Playback FPS" for speed control

## Documentation

See [`VISER_USAGE_GUIDE.md`](./VISER_USAGE_GUIDE.md) for detailed documentation on:
- Camera controls
- Joint control interface
- IK control interface  
- Trajectory playback
- Integration examples

## Examples

### Basic Visualization
```bash
# Static scene with default settings
python get_started/viser/viser_demo.py
```

### Interactive Control
```bash
# Joint control with different simulator
python get_started/viser/viser_demo.py --sim genesis --enable-joint-control

# IK control with different robot
python get_started/viser/viser_demo.py --robot ur5e --enable-ik-control
```

### Trajectory Playback
```bash
# Load and play trajectory
python get_started/viser/viser_demo.py --enable-trajectory \
    --trajectory-path roboverse_data/trajs/libero/pick_up_the_butter_and_place_it_in_the_basket/v2/franka_v2.pkl.gz
```

### Dynamic Simulation
```bash
# Run dynamic simulation and save video
python get_started/viser/viser_demo.py --dynamic --save-video
```

## Troubleshooting

### Port Already in Use
If port 8080 is occupied, the server will automatically try alternative ports.

### URDF Files Not Found
URDF files and meshes are automatically downloaded from HuggingFace if not found locally.

### IK Control Not Working
Ensure you have the IK solver installed:
- For pyroki: `pip install pyroki`
- For curobo: Follow curobo installation instructions

## Notes

- The demo uses headless mode by default (viser for visualization, not simulator's viewer)
- All control modes preserve the initial robot pose defined in the demo script
- Reset functionality in both Joint Control and IK Control returns to the demo-defined initial pose
- Multiple control modes can be enabled simultaneously
