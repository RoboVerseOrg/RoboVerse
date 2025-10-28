# 4. Motion Planning
In this tutorial, we will show you how to use MetaSim to plan a motion for a robot.
Note here, we use the `curobo` package to plan the motion. If you haven't installed it, please refer to our [curobo installation guide](http://127.0.0.1:8000/metasim/get_started/advanced_installation/curobo.html).

## Common Usage

```bash
python get_started/4_motion_planning.py  --sim <simulator>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will plan a motion for a robot and it will automatically record a video.


#### IsaacSim
```bash
python get_started/4_motion_planning.py  --sim isaacsim
```

#### Isaac Gym
```bash
python get_started/4_motion_planning.py  --sim isaacgym
```

#### Mujoco
```bash
# For mac users, replace python with mjpython.
python get_started/4_motion_planning.py  --sim mujoco --headless
```
Note that we find the `non-headless` mode of Mujoco is not stable. So we recommend using the `headless` mode.


#### Genesis
```bash
python get_started/4_motion_planning.py  --sim genesis
```
Note that we find the `headless` mode of Genesis is not stable. So we recommend using the `non-headless` mode.

#### Sapien
```bash
python get_started/4_motion_planning.py  --sim sapien3
```

#### Pybullet
```bash
python get_started/4_motion_planning.py  --sim pybullet
```

You will get the following videos:

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">IsaacSim</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_mujoco.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Mujoco</p>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_genesis.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Genesis</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_sapien3.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Sapien</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_pybullet.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">PyBullet</p>
        </div>
    </div>

</div>



## IK Solver Usage

MetaSim provides a unified IK solver that supports both `curobo` and `pyroki` backends for end-effector control. The IK solver is designed to work with multiple environments simultaneously and provides two main functions for end-effector control.

### Backend Options

The IK solver supports two backends:

- **`curobo`**: High-performance CUDA-accelerated IK solver with collision checking
- **`pyroki`**: JAX-based differentiable IK solver (default)


#### 1. `solve_ik_batch()` - Compute Arm Joint Positions

This function solves inverse kinematics for a batch of target end-effector poses, returning only the arm joint positions (excluding finger/gripper joints).

```python
from metasim.utils.ik_solver import setup_ik_solver

# Initialize IK solver
ik_solver = setup_ik_solver(robot_cfg, solver="pyroki")  # or "curobo"
# IK solver expects original joint order, but state uses alphabetical order
reorder_idx = handler.get_joint_reindex(scenario.robots[0].name)
inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]
# Solve IK for multiple environments
q_solution, ik_success = ik_solver.solve_ik_batch(
    ee_pos_target=target_positions,    # (B, 3) - target EE positions
    ee_quat_target=target_quaternions, # (B, 4) - target EE quaternions (wxyz)
    seed_q=curr_robot_q         # (B, n_dof) - seed configs (required for curobo)
)

# q_solution: (B, n_dof_ik) - arm joint positions only
# ik_success: (B,) - boolean mask indicating successful IK solutions, pyroki does not need this
```

#### 2. `compose_joint_action()` - Combine Arm + Gripper


This function combines the arm joint positions from IK with gripper positions to create the complete joint command. It can return either a tensor or action dictionaries.

```python
# Convert binary gripper command to joint widths
gripper_widths = process_gripper_command(
    gripper_binary=gripper_open_close,  # (B,) or (B, 1) - binary gripper state
    robot_cfg=robot_cfg,
    device=device
)

# Option 1: Return tensor in alphabetical order (default)
actions_tensor = ik_solver.compose_joint_action(
    q_solution=q_solution,           # (B, n_dof_ik) - arm joint positions from IK
    gripper_widths=gripper_widths,   # (B, ee_n_dof) - gripper joint positions
    current_q=current_joint_state,   # (B, n_robot_dof) - optional current state
    return_dict=False                # Default: return tensor
)
# q_full: (B, n_robot_dof) - complete joint command in alphabetical order

# Option 2: Return action dictionaries directly
actions_dict = ik_solver.compose_joint_action(
    q_solution=q_solution,
    gripper_widths=gripper_widths,
    current_q=current_joint_state,
    return_dict=True                 # Return action dictionaries
)
# actions: list of action dictionaries for env execution
```

**Note on Joint Ordering:**
- **Tensor output** (`return_dict=False`): Joints are ordered alphabetically, including end-effector joints
- **Dictionary output** (`return_dict=True`): Joints maintain the original dictionary order from robot configuration

**Example for Franka robot:**
```python
# Tensor order (alphabetical):
# [finger1, finger2, joint1, joint2, joint3, joint4, joint5, joint6, joint7]

# Dictionary order (original, with keys):
# [joint1, joint2, joint3, joint4, joint5, joint6, joint7, finger1, finger2]
```

### Complete End-Effector Control Example

```python
import torch
from metasim.utils.ik_solver import setup_ik_solver, process_gripper_command

# Setup
ik_solver = setup_ik_solver(robot_cfg, solver="pyroki")
num_envs = 4
device = torch.device("cuda")

# Target end-effector poses
target_positions = torch.randn(num_envs, 3, device=device)  # (B, 3)
target_quaternions = torch.randn(num_envs, 4, device=device)  # (B, 4) wxyz
gripper_commands = torch.randint(0, 2, (num_envs,), device=device)  # (B,) binary

# Step 1: Solve IK for arm joints
ik_solver = setup_ik_solver(robot_cfg, solver="pyroki")  # or "curobo"
# IK solver expects original joint order, but state uses alphabetical order
reorder_idx = handler.get_joint_reindex(scenario.robots[0].name)
inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]

q_arm, ik_success = ik_solver.solve_ik_batch(
    ee_pos_target=target_positions,
    ee_quat_target=target_quaternions,
    seed_q=curr_robot_q 
)

# Step 2: Process gripper commands
gripper_widths = process_gripper_command(
    gripper_binary=gripper_commands,
    robot_cfg=robot_cfg,
    device=device
)

# Step 3: Compose full joint command and create actions directly
actions = ik_solver.compose_joint_action(
    q_solution=q_arm,
    gripper_widths=gripper_widths,
    return_dict=True  # Return action dictionaries directly
)

```

### Backend-Specific Notes

**Curobo Backend:**
- Better performance for parallel environments
- Requires CUDA installation

**Pyroki Backend (Default):**
- JAX-based differentiable solver
- Lighter weight installation
- Good for research and prototyping

This IK solver design separates arm control (IK solving) from gripper control, making it easy to implement end-effector-based manipulation tasks across multiple environments.