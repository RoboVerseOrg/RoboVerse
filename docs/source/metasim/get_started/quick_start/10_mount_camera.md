# Tutorial 10: Mount Camera

**Objective**: Learn how to add and mount cameras to robots and objects for visual observations.

**What you'll learn**:
- Adding third-person view cameras
- Mounting ego-centric cameras on robot links
- Configuring camera resolution and intrinsics

**Prerequisites**: Completed [Tutorial 9: Config-Based Task](9_cfg_task)

**Estimated time**: 20 minutes

---

## Running the Tutorial

```bash
python get_started/10_mount_camera.py  --sim <simulator>
```
you can also render in the headless mode by adding `--headless` flag, only supported in `IsaacGym` and `IsaacSim`.

This script sets up a simulated scene and records video from two different camera perspectives
- Third-person view: A fixed camera placed above and behind the robot, observing the scene from a distance.
- Ego-centric view: A camera mounted directly on the robot’s torso_link, capturing the environment from the robot’s own perspective.


Both cameras use the PinholeCameraCfg configuration and record images at a resolution of 1024×1024.You can freely modify the camera pose (position and orientation) in the script to explore how it affects the rendered views.


You will get the following Videos:
---
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 42%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/10_mount_camera_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 42%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/10_mount_camera_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Lab</p>
        </div>
    </div>

</div>
