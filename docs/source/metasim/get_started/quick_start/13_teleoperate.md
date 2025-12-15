# 13. Teleoperate

## by Keyboard

### Dependencies

**Required dependencies:**

```bash
pip install pygame pynput opencv-python
```

### IK Solver Setup (Choose One)

**You must install one of the following IK solvers:**

#### Option 1: PyRoki (Default, Recommended)
```bash
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

#### Option 2: cuRobo (GPU-Accelerated, requires CUDA 11.8)
```bash
sudo apt install git-lfs
git submodule update --init --recursive
cd third_party/curobo
uv pip install -e . --no-build-isolation
```

**Note:** Install only **one** IK solver, not both.

### Optional: Viser 3D Visualization

```bash
pip install viser
```

Enable with `--enable-viser` flag (optional, not required for basic teleoperation).

### Play in Simulation

```bash
python scripts/advanced/teleop_keyboard.py --task close_box --robot franka --sim mujoco
```

**Note for mac users**: Use `--headless` flag.

Available tasks: `close_box`, `pick_cube`, `stack_cube`, `basketball_in_hoop`, and more.

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

**Checkpoint Control:**

| Key      | Action                                      |
|----------|---------------------------------------------|
| **B**    | Save checkpoint (save current state and trajectory) |
| **N**    | Restore to last checkpoint |

### Trajectory Recording

Trajectories are automatically saved to `teleop_trajs/` directory in v2 format. Use **V** to save episode, **R** to reset, **ESC** to exit and save all.

```bash
python scripts/advanced/teleop_keyboard.py --task close_box --save-every-n-steps 5 --traj-dir my_trajs
```

### Additional Options

```bash
# Run headless, choose simulator, control speed
python scripts/advanced/teleop_keyboard.py --task close_box --headless --sim mujoco --min-step-time 0.01
```

### Notes

- Install **one** IK solver (PyRoki recommended, cuRobo for GPU acceleration)
- Keyboard input is captured globally using `pynput`, window focus not required
- Trajectory data saved in v2 format compatible with replay scripts


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

**Network Configuration**: Server runs on `0.0.0.0:8765`. Connect to `ws://[PC_IP_ADDRESS]:8765` in the mobile app. Ensure phone and PC are on the same WiFi network.

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
- `BasketballInHoop`

### Instructions

- **Movement**: Phone buttons control gripper position (X/Y/Z axes)
- **Rotation**: Phone orientation controls end-effector rotation (uses accelerometer, magnetometer, and gyroscope)
- **Gripper**: Switch on device toggles gripper state (On: close, Off: open)

### Additional Notes:

- **Connection Setup**: Find PC's IP address: `ip addr show` (Linux) or `ipconfig` (Windows), then connect to `ws://[PC_IP_ADDRESS]:8765` in the mobile app
- Calibration may be needed the first time. Avoid strong electromagnetic sources near the phone.

---

## by Phone (Lerobot)

### Dependencies

```bash
pip install lerobot[phone]
```

### Play in Simulation

**For Android phone:**
```bash
python scripts/advanced/teleop_phone_lerobot.py --task stack_cube --robot franka --phone-os android
```

**For iOS phone:**
```bash
python scripts/advanced/teleop_phone_lerobot.py --task stack_cube --robot franka --phone-os ios
```

### Connection Instructions

1. Run the script - it will print a connection URL
2. **For Android**: Open the URL in your phone's browser
3. **For iOS**: Install and open the **HEBI Mobile I/O** app on your iPhone
4. Press the **Move/B1** button to start teleoperation

### Control Instructions

- **Movement**: Move your phone to control robot end-effector position
- **Rotation**: Rotate your phone to control robot end-effector orientation
- **Gripper**: Use buttons on the phone interface (Button A/B for Android, A3 slider for iOS)
- **Activation**: Hold **Move/B1** button to enable, release to pause

---
## by XR Headset

**System Requirements**: PICO 4 Ultra headset and Linux x86 PC (Ubuntu 22.04)

### Dependencies
You need to install `XRoboToolkit-PC-Service` on PC and `XRoboToolkit-PICO` app on your XR Headset. Follow the [XRoboToolkit Installation Instruction](./xrobotoolkit_instruction.md).

### Play in Simulation

```bash
python metasim/scripts/teleop_xr.py --task=PickCube
```

Available tasks: `PickCube`, `StackCube`, `CloseBox`, `BasketballInHoop`

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
- Start `XRoboToolkit-PC-Service` on PC and open `XRoboToolkit` app on headset