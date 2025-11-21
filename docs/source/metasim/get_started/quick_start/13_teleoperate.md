# 13. Teleoperate

## by Keyboard

### Dependencies

**Required dependencies:**

```bash
pip install pygame pynput opencv-python
```

### IK Solver Setup (Choose One)

The teleoperation system uses inverse kinematics (IK) solvers to control the robot. **You must install one of the following IK solvers** (choose based on your needs):

#### Option 1: PyRoki (Default, Recommended)
PyRoki is a modular and scalable robotics kinematics optimization library. It is the **default solver** and works with all simulators.

```bash
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

For more details, see the [PyRoki Installation Guide](../get_started/advanced_installation/pyroki.md).

#### Option 2: cuRobo (Advanced, GPU-Accelerated)
cuRobo is NVIDIA's GPU-accelerated motion planning library. It requires **CUDA 11.8** and provides faster IK solving for complex scenarios.

```bash
sudo apt install git-lfs
git submodule update --init --recursive
cd third_party/curobo
uv pip install -e . --no-build-isolation
```

For more details, see the [cuRobo Installation Guide](../get_started/advanced_installation/curobo.md).

**Note:** You only need to install **one** of the above IK solvers, not both.

### Optional: Viser 3D Visualization

If you want real-time 3D visualization in a web browser, you can optionally install Viser:

```bash
pip install viser
```

This is **completely optional** and not required for basic teleoperation.

### Play in Simulation

**Note for mac users**: Please run this task with normal python with the `--headless` tag.

**Basic usage (with default PyRoki solver):**
```bash
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --sim mujoco
```

**Specify IK solver (choose one):**
```bash
# Option 1: Using PyRoki (default, no flag needed)
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --ik-solver pyroki

# Option 2: Using cuRobo (requires cuRobo installation)
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --ik-solver curobo
```

**Optional: Enable Viser 3D visualization (requires viser installation):**
```bash
# Add --enable-viser flag to enable web-based 3D visualization
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --enable-viser

# Customize Viser port (default: 8080)
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --enable-viser --viser-port 9090
```

Available tasks include:
- `close_box`
- `pick_cube`
- `stack_cube`
- `basketball_in_hoop`
- And many more from the task registry

### Control Instructions

**Movement (Robot Base Coordinates):**

| Key    | Action                                   |
|--------|------------------------------------------|
| **↑** (UP) | Move end effector +X (forward) |
| **↓** (DOWN) | Move end effector -X (backward) |
| **←** (LEFT) | Move end effector +Y (left) |
| **→** (RIGHT) | Move end effector -Y (right) |
| **E**   | Move end effector +Z (up) |
| **D**   | Move end effector -Z (down) |

**Rotation (End Effector Local Coordinates):**

| Key    | Action                                    |
|--------|-------------------------------------------|
| **Q**  | Roll + (rotate around EE X-axis) |
| **W**  | Roll - (rotate around EE X-axis) |
| **A**  | Pitch + (rotate around EE Y-axis) |
| **S**  | Pitch - (rotate around EE Y-axis) |
| **Z**  | Yaw + (rotate around EE Z-axis) |
| **X**  | Yaw - (rotate around EE Z-axis) |

**Gripper Control:**

| Key      | Action                                      |
|----------|---------------------------------------------|
| **SPACE** | Close (hold) / Open (release) gripper |

**Episode Control:**

| Key      | Action                                      |
|----------|---------------------------------------------|
| **V**    | Complete current episode and save |
| **R**    | Reset and discard current episode |
| **ESC**  | Save all episodes and exit |

### Trajectory Recording

The teleoperation script automatically records trajectories in the v2 format with support for multiple episodes:

- Trajectories are saved to `teleop_trajs/` directory by default
- Each episode can be saved independently using the **V** key
- Press **R** to reset without saving the current episode
- On exit (ESC), all collected episodes are saved to a single file

Configure trajectory recording:
```bash
python scripts/advanced/teleop_keyboard.py \
    --task close_box \
    --save-traj \                    # Enable trajectory saving (default: True)
    --save-states \                  # Save full states, not just actions (default: True)
    --save-every-n-steps 5 \        # Downsample: save every N steps (default: 5)
    --traj-dir my_trajs              # Custom output directory
```

### Additional Options

**Display Settings:**
```bash
# Disable camera display (use pygame window only)
python scripts/advanced/teleop_keyboard.py --task close_box --no-display-camera

# Adjust camera display resolution
python scripts/advanced/teleop_keyboard.py --task close_box --display-width 1920 --display-height 1080
```

**Simulation Settings:**
```bash
# Choose simulator backend
python scripts/advanced/teleop_keyboard.py --task close_box --sim mujoco  # or genesis, isaacgym, etc.

# Run headless (no native renderer window)
python scripts/advanced/teleop_keyboard.py --task close_box --headless

# Control simulation speed
python scripts/advanced/teleop_keyboard.py --task close_box --min-step-time 0.01  # seconds per step
```

### Notes

**IK Solver Requirements:**
- **You must install ONE IK solver** (either PyRoki or cuRobo, not both)
- PyRoki is the default and recommended for most users
- cuRobo is faster but requires CUDA 11.8 and GPU

**Viser Visualization:**
- Viser is **completely optional** - the teleoperation works without it
- Only install Viser if you want web-based 3D visualization
- Enable with `--enable-viser` flag

**General:**
- The camera view displays real-time RGB observations from two cameras in split-screen mode
- Keyboard input is captured globally using `pynput`, so the window focus is not required
- The system uses IK solving to convert end-effector commands to joint positions
- Trajectory data is saved in v2 format compatible with replay scripts


---

## by Android Phone

### Dependencies

```bash
pip install websockets==11.0.3
```

**Note**: Use websockets version 11.0.3 for compatibility with the teleoperation system.

Additionally, you need to install the **teleoperation app** on your Android device. The app can be downloaded from[ \[App Store Link / GitHub Repository\].](https://github.com/co1one/teleop_app)

### Play in Simulation
```bash
python scripts/advanced/teleop_phone.py --task PickCube --num_envs 1
```

**Network Configuration**:
- The server runs on `0.0.0.0:8765` by default
- Use your PC's **local network IP address** (e.g., `192.168.1.100:8765`) in the mobile app
- Ensure your phone and PC are on the same WiFi network

**Quick IP Address Check**:
```bash
# Linux/macOS
ip addr show | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig | findstr "IPv4"
```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
- `BasketballInHoop`

### Instructions

The Android controller uses a combination of sensors and screen gestures to provide a unique and intuitive control experience.

#### Movement Controls (Gripper Coordinates):

- **Buttons** on the phone control the movement of the gripper in 3D space along the gripper’s coordinate system:
  - **Forward / Backward** (X-axis)
  - **Up / Down** (Y-axis)
  - **Left / Right** (Z-axis)

- **Switch** on the device toggles the gripper's state:
  - **On**: Close the gripper.
  - **Off**: Open the gripper.

#### Rotation Control (Phone Orientation):

The rotation of the phone itself is used to control the rotation of the robot's end effector. The Android device uses the following sensors to provide real-time rotation data:

1. **Accelerometer**: Measures the device’s linear acceleration and provides tilt information relative to the Earth's gravity.
2. **Magnetometer**: Detects the device’s orientation with respect to the Earth's magnetic field, helping determine its heading.
3. **Gyroscope**: Tracks the rotational velocity, allowing the app to track angular changes.

These sensors work together to provide a **rotation vector**, which represents the device's orientation in space using a **quaternion**. This quaternion avoids issues like gimbal lock and provides smooth rotational control.

#### Sensor Fusion for Control:

- The rotation vector from the phone provides the **pitch**, **yaw**, and **roll** controls for the robot’s end effector.
- Tilting the phone will control the pitch and yaw, while rotating it along the Z-axis adjusts the roll.
- The gripper's actions are toggled via the button switch, and the movement controls are directly tied to the phone's directional buttons.

### Additional Notes:

- Ensure the **Android app** is running and connected to the PC.
- **Connection Setup**: 
  - Find your PC's IP address: `ip addr show` (Linux) or `ipconfig` (Windows)
  - In the mobile app, connect to `ws://[PC_IP_ADDRESS]:8765`
  - Example: `ws://192.168.1.100:8765`
- Calibration may be needed the first time you use the phone to ensure the sensors are aligned with the robot's coordinate system.
- Ensure that the phone is not held too close to strong electromagnetic sources (e.g., motors, power lines, or other electronic devices) as they can interfere with the sensors' ability to accurately determine the "North" direction, which is crucial for rotation control.

---
## by XR Headset
Run a full XR-to-robot teleoperation sample on a **PICO 4 Ultra headset and a Linux x86 PC**. System Requirements:

- Linux x86 PC: Ubuntu 22.04

- PICO 4 Ultra: User OS >5.12. Currently supports [PICO 4 Ultra](https://www.picoxr.com/global/products/pico4-ultra) and [PICO 4 Ultra Enterprise](https://www.picoxr.com/global/products/pico4-ultra-enterprise).

### Dependencies
You need to install `XRoboToolkit-PC-Service` on PC and `XRoboToolkit-PICO` app on your XR Headset. Follow the [XRoboToolkit Installation Instruction](./xrobotoolkit_instruction.md).

### Play in Simulation
   - Run the XR teleoperation demo
      ```bash
      python metasim/scripts/teleop_xr.py --task=PickCube
      ```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
- `BasketballInHoop`

### Instructions

Movement (World Coordinates):

| Key    | Action                                   |
|--------|------------------------------------------|
| **Grip** | Hold Grip key to activate teleoperation |

Gripper:

| Key      | Action                                      |
|----------|---------------------------------------------|
| **Target** | Close (hold) / Open (release) the gripper  |

Simulation Control:

| Key      | Action                                      |
|----------|---------------------------------------------|
| **A**  | Toggle start/stop sending data from headset|
| **B**  | Quit the simulation and exit               |

### Additional Notes:
- Connect robot PC and Pico 4 Ultra under the same network
- On robot PC, double click app icon of `XRoboToolkit-PC-Service` or run service `/opt/apps/roboticsservice/runService.sh`
- Open app `XRoboToolkit` on the Pico headset. Details of the Unity app can be found in the [Unity source repo](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client).