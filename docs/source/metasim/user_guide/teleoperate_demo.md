# Teleoperate

## by Keyboard

### Dependencies

Before using the keyboard controller, install the required Python library:

```bash
pip install pygame
```

### Play in Simulation

```bash
python metasim/scripts/teleop_keyboard.py --task PickCube --num_envs 1
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
| **UP** | Move +X (Move the end effector forward along the X-axis) |
| **DOWN** | Move -X (Move the end effector backward along the X-axis) |
| **LEFT** | Move +Y (Move the end effector left along the Y-axis) |
| **RIGHT** | Move -Y (Move the end effector right along the Y-axis) |
| **e**   | Move +Z (Move the end effector up along the Z-axis) |
| **d**   | Move -Z (Move the end effector down along the Z-axis) |

Rotation (End Effector Coordinates):

| Key    | Action                                    |
|--------|-------------------------------------------|
| **q**  | Roll + (Rotate the end effector around its X-axis clockwise) |
| **w**  | Roll - (Rotate the end effector around its X-axis counterclockwise) |
| **a**  | Pitch + (Rotate the end effector around its Y-axis upwards) |
| **s**  | Pitch - (Rotate the end effector around its Y-axis downwards) |
| **z**  | Yaw + (Rotate the end effector around its Z-axis counterclockwise) |
| **x**  | Yaw - (Rotate the end effector around its Z-axis clockwise) |

Gripper:

| Key      | Action                                      |
|----------|---------------------------------------------|
| **Space** | Close (hold) / Open (release) the gripper  |

Simulation Control:

| Key      | Action                                      |
|----------|---------------------------------------------|
| **ESC**  | Quit the simulation and exit               |

### Additional Notes:

- After successfully running the simulation, a **pygame window** will appear. The window will display the instructions for keyboard control. Make sure that the **cursor is focused on the pygame window** for the controls to work. This design helps prevent conflicts with hotkeys or other functionalities in the simulation's main visualization window.


---

## by Android Phone

### Dependencies

```bash
pip install websockets
```

Additionally, you need to install the **teleoperation app** on your Android device. The app can be downloaded from [App Store Link / GitHub Repository].

### Play in Simulation
```bash
python metasim/scripts/teleop_keyboard.py --task PickCube --num_envs 1
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

- Ensure the **Android app** is running and connected to the pc.
- Calibration may be needed the first time you use the phone to ensure the sensors are aligned with the robot’s coordinate system.
- Ensure that the phone is not hold too close to strong electromagnetic sources (e.g., motors, power lines, or other electronic devices) as they can interfere with the sensors' ability to accurately determine the "North" direction, which is crucial for rotation control.

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
