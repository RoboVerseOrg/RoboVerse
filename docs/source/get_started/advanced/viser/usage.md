# Viser Visualization

## Prerequisites

Before using Viser visualization, ensure you have the required dependencies installed:

```bash
pip install viser            # Core dependencies only
pip install viser[examples]  # To include example dependencies
```

## Basic Setup

The easiest way to add real-time 3D visualization is to wrap your environment with `TaskViserWrapper`:

```python
from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

# Basic usage - just wrap your environment
env = TaskViserWrapper(env)

# That's it! Your environment now has real-time 3D visualization
```

For more control over visualization settings:

```python
# Custom port and update frequency
env = TaskViserWrapper(env, port=8080, update_freq=30)
```

## Accessing the Visualization

Once your environment is wrapped with `TaskViserWrapper`, the Viser server will automatically start. Access the interactive 3D visualization interface by opening your web browser and navigating to:

```
http://localhost:8080/
```

If you specified a custom port (e.g., `port=8888`), use that port instead:

```
http://localhost:8888/
```

The visualization interface will load in your browser, providing real-time 3D rendering and interactive controls for your robot simulation.

> **Note**: The Viser server must be running (i.e., your Python script with the wrapped environment must be active) for the browser interface to work.


## Advance : Interactive Features

Viser provides comprehensive interactive 3D visualization and robot control capabilities through an intuitive web interface. The system supports multiple control modes and advanced visualization features.

## Core Features

### 1. Real-time 3D Visualization
- **Live Scene Rendering**: View your robot and environment in real-time 3D
- **Multi-camera Support**: Flexible camera positioning and controls
- **Performance Optimized**: Efficient rendering with configurable update rates

### 2. Interactive Robot Control
- **Joint Control**: Direct manipulation of individual robot joints via GUI sliders
- **End-Effector Control**: Intuitive control of end-effector position and orientation
- **Real-time Feedback**: Immediate visual feedback for all control inputs

### 3. Trajectory Management
- **Trajectory Playback**: Load and play back recorded robot trajectories
- **Speed Control**: Adjustable playback speed with scrubbing capabilities
- **Multi-trajectory Support**: Switch between different recorded demonstrations

### 4. Advanced Visualization Tools
- **Coordinate Frames**: Visual reference frames for better spatial understanding
- **Contact Visualization**: View contact forces and interactions
- **State Recording**: Capture and save simulation states for analysis


## Interactive Control Interface

### Camera Controls
- **Rotate**: Left mouse drag
- **Pan**: Right mouse drag or Shift+Left drag
- **Zoom**: Scroll wheel
- **Preset Views**: Top, Side, Front buttons
- **Screenshot**: Capture button
- **Video Recording**: Start/Stop recording

## Robot Control Features

### Joint Control Mode
1. Select robot from dropdown menu
2. Click "Setup Joint Control" to enable sliders
3. Use individual joint sliders to control each degree of freedom
4. Click "Reset Joints" to return to initial pose
5. Click "Clear Joint Control" to remove GUI controls

### End-Effector Control Mode
1. Click "Setup IK Control" to enable end-effector controls
2. Adjust target position sliders (X, Y, Z coordinates)
3. Adjust target orientation sliders (Quaternion W, X, Y, Z)
4. Visual markers show target position (red sphere + RGB coordinate axes)
5. Click "Solve & Apply IK" to move robot to target pose
6. Click "Reset Robot Joints" to return to initial configuration
7. Click "Reset Target" to reset target markers to default

### Trajectory Playback Mode
1. Load trajectory via file path parameter
2. Click "Update Robot List" to refresh available robots
3. Select robot and trajectory from dropdown menus
4. Click "Set Current Trajectory" to load the selected trajectory
5. Use Play/Pause/Stop controls to control playback
6. Drag timeline slider to scrub through trajectory
7. Adjust "Playback FPS" slider for speed control

## Advanced Features

### Multi-Robot Support
- Visualize and control multiple robots simultaneously
- Individual control interfaces for each robot
- Synchronized trajectory playback across multiple robots

### Performance Optimization
- Configurable update frequency to balance visualization quality and performance
- Automatic camera view refresh for smooth interaction
- Efficient state extraction and visualization updates


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

- Viser uses headless mode by default (viser for visualization, not simulator's viewer)
- All control modes preserve the initial robot pose defined in your environment
- Reset functionality returns to the environment's initial configuration
- Multiple control modes can be enabled simultaneously
- The system automatically handles different environment types (gym-style and standard RLTaskEnv)
