from __future__ import annotations

import copy
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Warning: matplotlib or numpy not installed. Plotting functionality will be disabled.")
    plt = None
    np = None

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class
from metasim.utils.math import quat_rotate_inverse, euler_xyz_from_quat

from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes
from roboverse_learn.rl.unitree_rl.helper import (get_args, make_objects, get_log_dir,
                                                  make_robots, set_seed, get_load_path,
                                                  PolicyExporterLSTM, export_policy_as_jit,
                                                  get_export_jit_path, get_indices_from_substring)
from roboverse_learn.rl.unitree_rl.runners import EnvWrapperTypes, MasterRunner

def prepare(args):
    task_cls = get_task_class(args.task)
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    overrides = {
        "num_envs": args.num_envs,
        "simulator": args.sim,
        "headless": args.headless,
    }

    if args.robots:
        overrides["robots"] = make_robots(args.robots)
        overrides["cameras"] = [
            camera
            for robot in overrides["robots"]
            if hasattr(robot, "cameras")
            for camera in getattr(robot, "cameras", [])
        ]

    if args.objects:
        overrides["objects"] = make_objects(args.objects)

    scenario.update(**overrides)

    device = "cpu" if args.sim == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    master_runner = MasterRunner(
        task_cls=task_cls,
        scenario=scenario,
        log_path=args.resume,
        lib_name="rsl_rl",
        device=device,
    )

    return master_runner

def play(args):
    master_runner = prepare(args)
    name_0 = list(master_runner.runners.keys())[0]
    if args.resume:
        if args.jit_load:
            log_dir = get_log_dir(task_name=master_runner.task_name, now=args.resume)
            policy_0 = torch.jit.load(get_load_path(load_root=log_dir, checkpoint=args.checkpoint))
        else:
            policys = master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
            policy_0 = policys[name_0]
    else:
        raise ValueError("Please provide the resume dir for eval policy.")

    runner_0 = master_runner.runners[name_0]
    env_0: EnvTypes = runner_0.env
    envwrapper_0: EnvWrapperTypes = runner_0.env_wrapper
    cfg_0 = env_0.cfg

    cfg_0.curriculum.enabled = False
    cfg_0.commands.resampling_time = 1e6  # effectively disable command changes

    # export jit policy
    export_jit_path = get_export_jit_path(get_log_dir(task_name=master_runner.task_name, now=args.resume), master_runner.scenario)
    actor_critic = runner_0.runner.alg.policy
    if hasattr(actor_critic, "memory_a"):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(export_jit_path)
    else:
        export_policy_as_jit(actor_critic.actor, export_jit_path)
    print("Exported policy as jit script to: ", export_jit_path)

    # unenable noise and randomization for eval

    env_0.reset()
    obs, _, _, _, _ = env_0.step(torch.zeros(env_0.num_envs, env_0.num_actions, device=env_0.device))
    obs = envwrapper_0.get_observations()

    # Data collection variables
    plot_time = 5.0  # Start plotting after collecting 5 seconds of data
    current_time = 0.0
    step_dt = env_0.step_dt
    plot_done = False

    # Data storage
    time_data = []
    cmd_vel_data = []  # [x, y, z]
    actual_vel_data = []  # [x, y, z]
    joint_torque_data = []  # [num_joints]
    joint_pos_actual_data = []  # [num_joints]
    joint_pos_target_data = []  # [num_joints]
    base_orientation_data = []  # [roll, pitch, yaw]
    left_foot_contact_data = []
    right_foot_contact_data = []
    left_foot_height_data = []
    right_foot_height_data = []
    left_foot_force_data = []  # Contact force magnitude
    right_foot_force_data = []  # Contact force magnitude
    left_foot_force_xyz_data = []  # Contact force [x, y, z]
    right_foot_force_xyz_data = []  # Contact force [x, y, z]
    observation_data = []  # All observations

    # Get foot indices
    body_names = env_0.handler.get_body_names(env_0.name, sort=True)
    left_foot_indices = get_indices_from_substring(".*left_toe_roll.*", body_names)
    right_foot_indices = get_indices_from_substring(".*right_toe_roll.*", body_names)

    # Get joint names
    joint_names = env_0.handler.get_joint_names(env_0.name, sort=True)

    for i in range(1000000):
        # set fixed command
        env_0.commands_manager.value[:, 0] = 1.0
        env_0.commands_manager.value[:, 1] = 0.0
        env_0.commands_manager.value[:, 2] = 0.0
        actions = policy_0(obs)

        # Collect observation data before step
        # Get observation directly from environment to avoid TensorDict indexing issues
        try:
            obs_policy = env_0.obs_buf[0].detach().cpu().numpy()
            observation_data.append(obs_policy)
        except Exception as e:
            # Fallback: try to extract from obs TensorDict
            try:
                from tensordict import TensorDict
                if isinstance(obs, TensorDict):
                    obs_policy_tensor = obs.get('policy')
                    if obs_policy_tensor is not None:
                        # Handle tensor: if 2D with batch, take first or squeeze; if 1D, use directly
                        if isinstance(obs_policy_tensor, torch.Tensor):
                            if len(obs_policy_tensor.shape) == 2:
                                if obs_policy_tensor.shape[0] == 1:
                                    obs_policy = obs_policy_tensor.squeeze(0).detach().cpu().numpy()
                                else:
                                    obs_policy = obs_policy_tensor[0].detach().cpu().numpy()
                            else:
                                obs_policy = obs_policy_tensor.detach().cpu().numpy()
                        else:
                            obs_policy = obs_policy_tensor
                        observation_data.append(obs_policy)
                    else:
                        print(f"Warning: Could not find 'policy' key in observation TensorDict")
                else:
                    print(f"Warning: Observation is not a TensorDict: {type(obs)}")
            except Exception as e2:
                print(f"Warning: Could not collect observation data: {e2}")
                pass

        obs, _, _, _ = envwrapper_0.step(actions)

        # Update simulation time
        current_time += step_dt

        # Collect data from the beginning
        # Get current state
        env_states = env_0.handler.get_states()
        robot_state = env_states.robots[env_0.name]

        # Collect command velocity
        cmd_vel = env_0.commands_manager.value[0].detach().cpu().numpy()  # [x, y, z]
        cmd_vel_data.append(cmd_vel)

        # Collect actual velocity (in robot local frame)
        base_quat = robot_state.root_state[0, 3:7]
        base_lin_vel_world = robot_state.root_state[0, 7:10]
        base_lin_vel_local = quat_rotate_inverse(base_quat.unsqueeze(0), base_lin_vel_world.unsqueeze(0))[0]
        actual_vel_data.append(base_lin_vel_local.detach().cpu().numpy())

        # Collect actual joint torques
        # Compute actual torques using the same method as the environment
        if hasattr(env_0, '_compute_effort') and env_0.manual_pd_on:
            # For effort control mode, compute torques using PD controller
            processed_actions = (actions[0:1] * env_0.action_scale)
            if env_0.cfg.control.action_offset:
                processed_actions = processed_actions + env_0.default_dof_pos.unsqueeze(0)
            processed_actions = processed_actions.clip(-env_0.action_clip, env_0.action_clip)
            computed_effort = env_0._compute_effort(processed_actions, env_states)
            joint_torques = computed_effort[0].detach().cpu().numpy()
        elif hasattr(env_0.handler, 'gym') and hasattr(env_0.handler, '_dof_force'):
            # For position control mode, try to get measured forces from simulator
            env_0.handler.gym.refresh_dof_force_tensor(env_0.handler.sim)
            joint_ids_reindex = env_0.handler._get_joint_ids_reindex(env_0.name)
            joint_torques = env_0.handler._dof_force.view(env_0.handler.num_envs, -1)[0, joint_ids_reindex].detach().cpu().numpy()
        else:
            # Fallback to using joint_effort_target
            joint_torques = robot_state.joint_effort_target[0].detach().cpu().numpy()
        joint_torque_data.append(joint_torques)

        # Collect joint positions
        joint_pos_actual = robot_state.joint_pos[0].detach().cpu().numpy()
        joint_pos_actual_data.append(joint_pos_actual)

        # Calculate target position: action * scale + default
        action_scaled = actions[0].detach().cpu().numpy() * env_0.action_scale
        # Add default based on action_offset configuration
        if env_0.cfg.control.action_offset:
            joint_pos_target = action_scaled + env_0.default_dof_pos.detach().cpu().numpy()
        else:
            joint_pos_target = action_scaled
        joint_pos_target_data.append(joint_pos_target)

        # Collect base orientation (base_euler_xyz: roll, pitch, yaw)
        roll, pitch, yaw = euler_xyz_from_quat(base_quat.unsqueeze(0))
        base_euler_xyz = torch.stack([roll, pitch, yaw], dim=-1)[0]
        base_orientation_data.append(base_euler_xyz.detach().cpu().numpy())

        # Collect foot contact information
        if "contact_forces" in env_states.extras:
            contact_forces = env_states.extras["contact_forces"][env_0.name]
            # Get left and right foot contact forces
            if len(left_foot_indices) > 0:
                left_foot_force_vec = contact_forces.contact_forces[0, left_foot_indices[0], :]
                left_foot_force = left_foot_force_vec.norm().item()
                left_foot_contact_data.append(1.0 if left_foot_force > 1.0 else 0.0)
                left_foot_force_data.append(left_foot_force)
                left_foot_force_xyz_data.append(left_foot_force_vec.detach().cpu().numpy())
                # Get left foot height
                left_foot_pos = robot_state.body_state[0, left_foot_indices[0], 2].item()
                left_foot_height_data.append(left_foot_pos)
            else:
                left_foot_contact_data.append(0.0)
                left_foot_height_data.append(0.0)
                left_foot_force_data.append(0.0)
                left_foot_force_xyz_data.append(np.array([0.0, 0.0, 0.0]))

            if len(right_foot_indices) > 0:
                right_foot_force_vec = contact_forces.contact_forces[0, right_foot_indices[0], :]
                right_foot_force = right_foot_force_vec.norm().item()
                right_foot_contact_data.append(1.0 if right_foot_force > 1.0 else 0.0)
                right_foot_force_data.append(right_foot_force)
                right_foot_force_xyz_data.append(right_foot_force_vec.detach().cpu().numpy())
                # Get right foot height
                right_foot_pos = robot_state.body_state[0, right_foot_indices[0], 2].item()
                right_foot_height_data.append(right_foot_pos)
            else:
                right_foot_contact_data.append(0.0)
                right_foot_height_data.append(0.0)
                right_foot_force_data.append(0.0)
                right_foot_force_xyz_data.append(np.array([0.0, 0.0, 0.0]))
        else:
            left_foot_contact_data.append(0.0)
            right_foot_contact_data.append(0.0)
            left_foot_height_data.append(0.0)
            right_foot_height_data.append(0.0)
            left_foot_force_data.append(0.0)
            right_foot_force_data.append(0.0)
            left_foot_force_xyz_data.append(np.array([0.0, 0.0, 0.0]))
            right_foot_force_xyz_data.append(np.array([0.0, 0.0, 0.0]))

        # Record time (starting from 0)
        time_data.append(current_time)

        # Start plotting after collecting 5 seconds of data
        if current_time >= plot_time and not plot_done:
            print(f"Collected {current_time:.2f} seconds of data, starting to plot analysis figures...")
            plot_done = True

            # Plot figures
            if len(time_data) > 0 and plt is not None and np is not None:
                log_dir = get_log_dir(task_name=master_runner.task_name, now=args.resume)
                # Get simulator name from scenario
                simulator_name = master_runner.scenario.simulator or "unknown"
                plot_dir = os.path.join(log_dir, "eval_plots", simulator_name)
                os.makedirs(plot_dir, exist_ok=True)

                time_array = np.array(time_data)

                # 1. Plot command velocity vs actual velocity
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                cmd_vel_array = np.array(cmd_vel_data)
                actual_vel_array = np.array(actual_vel_data)

                axes[0].plot(time_array, cmd_vel_array[:, 0], label='Command X', linewidth=2)
                axes[0].plot(time_array, actual_vel_array[:, 0], label='Actual X', linewidth=2, linestyle='--')
                axes[0].set_ylabel('Velocity X (m/s)')
                axes[0].set_title('Velocity Comparison - X')
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(time_array, cmd_vel_array[:, 1], label='Command Y', linewidth=2)
                axes[1].plot(time_array, actual_vel_array[:, 1], label='Actual Y', linewidth=2, linestyle='--')
                axes[1].set_ylabel('Velocity Y (m/s)')
                axes[1].set_title('Velocity Comparison - Y')
                axes[1].legend()
                axes[1].grid(True)

                axes[2].plot(time_array, cmd_vel_array[:, 2], label='Command Z', linewidth=2)
                axes[2].plot(time_array, actual_vel_array[:, 2], label='Actual Z', linewidth=2, linestyle='--')
                axes[2].set_xlabel('Time (s)')
                axes[2].set_ylabel('Angular Velocity Z (rad/s)')
                axes[2].set_title('Angular Velocity Comparison - Z')
                axes[2].legend()
                axes[2].grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "velocity_comparison.png"), dpi=150)
                plt.close()

                # 2. Plot joint torques (n x 2 layout: left leg on left, right leg on right)
                joint_torque_array = np.array(joint_torque_data)
                num_joints = joint_torque_array.shape[1]

                # Identify left and right leg joint indices
                left_leg_indices = [i for i, name in enumerate(joint_names) if 'left' in name.lower()]
                right_leg_indices = [i for i, name in enumerate(joint_names) if 'right' in name.lower()]

                # Ensure we have equal number of joints per leg
                n_per_leg = len(left_leg_indices)
                if len(right_leg_indices) != n_per_leg:
                    # Fallback: split in half
                    n_per_leg = num_joints // 2
                    left_leg_indices = list(range(n_per_leg))
                    right_leg_indices = list(range(n_per_leg, num_joints))

                # Create n x 2 subplot layout
                fig, axes = plt.subplots(n_per_leg, 2, figsize=(12, 3 * n_per_leg))
                if n_per_leg == 1:
                    axes = axes.reshape(1, -1)

                # Plot left leg joints on the left column
                for row, joint_idx in enumerate(left_leg_indices):
                    axes[row, 0].plot(time_array, joint_torque_array[:, joint_idx], linewidth=2)
                    axes[row, 0].set_title(f'{joint_names[joint_idx]}')
                    axes[row, 0].set_ylabel('Torque (Nm)')
                    axes[row, 0].grid(True)
                    if row == n_per_leg - 1:
                        axes[row, 0].set_xlabel('Time (s)')

                # Plot right leg joints on the right column
                for row, joint_idx in enumerate(right_leg_indices):
                    axes[row, 1].plot(time_array, joint_torque_array[:, joint_idx], linewidth=2)
                    axes[row, 1].set_title(f'{joint_names[joint_idx]}')
                    axes[row, 1].set_ylabel('Torque (Nm)')
                    axes[row, 1].grid(True)
                    if row == n_per_leg - 1:
                        axes[row, 1].set_xlabel('Time (s)')

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "joint_torques.png"), dpi=150)
                plt.close()

                # 3. Plot left and right foot contact phase (height plot)
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                left_foot_contact_array = np.array(left_foot_contact_data)
                right_foot_contact_array = np.array(right_foot_contact_data)
                left_foot_height_array = np.array(left_foot_height_data)
                right_foot_height_array = np.array(right_foot_height_data)

                # Left foot
                ax1 = axes[0]
                ax1_twin = ax1.twinx()
                ax1.fill_between(time_array, 0, left_foot_contact_array, alpha=0.3, color='red', label='Contact')
                ax1_twin.plot(time_array, left_foot_height_array, color='blue', linewidth=2, label='Height')
                ax1.set_ylabel('Contact (0/1)', color='red')
                ax1_twin.set_ylabel('Height (m)', color='blue')
                ax1.set_title('Left Foot Contact State and Height')
                ax1.set_ylim([-0.1, 1.1])
                ax1.legend(loc='upper left')
                ax1_twin.legend(loc='upper right')
                ax1.grid(True)

                # Right foot
                ax2 = axes[1]
                ax2_twin = ax2.twinx()
                ax2.fill_between(time_array, 0, right_foot_contact_array, alpha=0.3, color='red', label='Contact')
                ax2_twin.plot(time_array, right_foot_height_array, color='blue', linewidth=2, label='Height')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Contact (0/1)', color='red')
                ax2_twin.set_ylabel('Height (m)', color='blue')
                ax2.set_title('Right Foot Contact State and Height')
                ax2.set_ylim([-0.1, 1.1])
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "foot_contact_phase.png"), dpi=150)
                plt.close()

                # 4. Plot joint actual positions and target positions (n x 2 layout: left leg on left, right leg on right)
                joint_pos_actual_array = np.array(joint_pos_actual_data)
                joint_pos_target_array = np.array(joint_pos_target_data)
                default_dof_pos_array = env_0.default_dof_pos.detach().cpu().numpy()

                # Identify left and right leg joint indices (same as torque plot)
                left_leg_indices_pos = [i for i, name in enumerate(joint_names) if 'left' in name.lower()]
                right_leg_indices_pos = [i for i, name in enumerate(joint_names) if 'right' in name.lower()]

                # Ensure we have equal number of joints per leg
                n_per_leg_pos = len(left_leg_indices_pos)
                if len(right_leg_indices_pos) != n_per_leg_pos:
                    # Fallback: split in half
                    n_per_leg_pos = num_joints // 2
                    left_leg_indices_pos = list(range(n_per_leg_pos))
                    right_leg_indices_pos = list(range(n_per_leg_pos, num_joints))

                # Create n x 2 subplot layout
                fig, axes = plt.subplots(n_per_leg_pos, 2, figsize=(12, 3 * n_per_leg_pos))
                if n_per_leg_pos == 1:
                    axes = axes.reshape(1, -1)

                # Plot left leg joints on the left column
                for row, joint_idx in enumerate(left_leg_indices_pos):
                    axes[row, 0].plot(time_array, joint_pos_actual_array[:, joint_idx], label='Actual', linewidth=2)
                    axes[row, 0].plot(time_array, joint_pos_target_array[:, joint_idx], label='Target', linewidth=2, linestyle='--')
                    axes[row, 0].axhline(y=default_dof_pos_array[joint_idx], color='gray', linestyle=':', linewidth=2, label='Default')
                    axes[row, 0].set_title(f'{joint_names[joint_idx]}')
                    axes[row, 0].set_ylabel('Position (rad)')
                    axes[row, 0].legend()
                    axes[row, 0].grid(True)
                    if row == n_per_leg_pos - 1:
                        axes[row, 0].set_xlabel('Time (s)')

                # Plot right leg joints on the right column
                for row, joint_idx in enumerate(right_leg_indices_pos):
                    axes[row, 1].plot(time_array, joint_pos_actual_array[:, joint_idx], label='Actual', linewidth=2)
                    axes[row, 1].plot(time_array, joint_pos_target_array[:, joint_idx], label='Target', linewidth=2, linestyle='--')
                    axes[row, 1].axhline(y=default_dof_pos_array[joint_idx], color='gray', linestyle=':', linewidth=2, label='Default')
                    axes[row, 1].set_title(f'{joint_names[joint_idx]}')
                    axes[row, 1].set_ylabel('Position (rad)')
                    axes[row, 1].legend()
                    axes[row, 1].grid(True)
                    if row == n_per_leg_pos - 1:
                        axes[row, 1].set_xlabel('Time (s)')

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "joint_positions.png"), dpi=150)
                plt.close()

                # 5. Plot base orientation
                base_orientation_array = np.array(base_orientation_data)
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))

                axes[0].plot(time_array, np.degrees(base_orientation_array[:, 0]), linewidth=2)
                axes[0].set_ylabel('Roll (deg)')
                axes[0].set_title('Base Orientation - Roll')
                axes[0].grid(True)

                axes[1].plot(time_array, np.degrees(base_orientation_array[:, 1]), linewidth=2)
                axes[1].set_ylabel('Pitch (deg)')
                axes[1].set_title('Base Orientation - Pitch')
                axes[1].grid(True)

                axes[2].plot(time_array, np.degrees(base_orientation_array[:, 2]), linewidth=2)
                axes[2].set_xlabel('Time (s)')
                axes[2].set_ylabel('Yaw (deg)')
                axes[2].set_title('Base Orientation - Yaw')
                axes[2].grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "base_orientation.png"), dpi=150)
                plt.close()

                # 6. Plot all observations
                observation_array = np.array(observation_data)
                num_obs_dims = observation_array.shape[1]

                # Calculate grid size for subplots (aim for roughly square grid)
                n_cols = int(np.ceil(np.sqrt(num_obs_dims)))
                n_rows = int(np.ceil(num_obs_dims / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)
                else:
                    axes = axes.flatten()

                for dim_idx in range(num_obs_dims):
                    row = dim_idx // n_cols
                    col = dim_idx % n_cols
                    if n_rows == 1 and n_cols == 1:
                        ax = axes
                    elif n_rows == 1:
                        ax = axes[col]
                    elif n_cols == 1:
                        ax = axes[row]
                    else:
                        ax = axes[dim_idx]

                    ax.plot(time_array, observation_array[:, dim_idx], linewidth=1.5)
                    ax.set_title(f'Obs Dim {dim_idx}', fontsize=8)
                    ax.set_ylabel('Value', fontsize=7)
                    ax.grid(True, alpha=0.3)
                    if row == n_rows - 1 or (n_rows == 1 and dim_idx == num_obs_dims - 1):
                        ax.set_xlabel('Time (s)', fontsize=7)

                # Hide unused subplots
                for dim_idx in range(num_obs_dims, n_rows * n_cols):
                    row = dim_idx // n_cols
                    col = dim_idx % n_cols
                    if n_rows == 1 and n_cols == 1:
                        ax = axes
                    elif n_rows == 1:
                        ax = axes[col]
                    elif n_cols == 1:
                        ax = axes[row]
                    else:
                        ax = axes[dim_idx]
                    ax.axis('off')

                plt.suptitle('All Observations Over Time', fontsize=12, y=0.995)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "observations.png"), dpi=150)
                plt.close()

                # 6. Plot foot contact forces
                left_foot_force_array = np.array(left_foot_force_data)
                right_foot_force_array = np.array(right_foot_force_data)
                left_foot_force_xyz_array = np.array(left_foot_force_xyz_data)
                right_foot_force_xyz_array = np.array(right_foot_force_xyz_data)

                fig, axes = plt.subplots(2, 2, figsize=(14, 8))

                # Top row: Contact force magnitude
                axes[0, 0].plot(time_array, left_foot_force_array, linewidth=2, color='blue', label='Left Foot')
                axes[0, 0].set_ylabel('Contact Force Magnitude (N)')
                axes[0, 0].set_title('Left Foot Contact Force')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

                axes[0, 1].plot(time_array, right_foot_force_array, linewidth=2, color='red', label='Right Foot')
                axes[0, 1].set_ylabel('Contact Force Magnitude (N)')
                axes[0, 1].set_title('Right Foot Contact Force')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

                # Bottom row: Contact force components (x, y, z)
                axes[1, 0].plot(time_array, left_foot_force_xyz_array[:, 0], linewidth=2, label='X', color='red')
                axes[1, 0].plot(time_array, left_foot_force_xyz_array[:, 1], linewidth=2, label='Y', color='green')
                axes[1, 0].plot(time_array, left_foot_force_xyz_array[:, 2], linewidth=2, label='Z', color='blue')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Contact Force Components (N)')
                axes[1, 0].set_title('Left Foot Contact Force Components')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

                axes[1, 1].plot(time_array, right_foot_force_xyz_array[:, 0], linewidth=2, label='X', color='red')
                axes[1, 1].plot(time_array, right_foot_force_xyz_array[:, 1], linewidth=2, label='Y', color='green')
                axes[1, 1].plot(time_array, right_foot_force_xyz_array[:, 2], linewidth=2, label='Z', color='blue')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Contact Force Components (N)')
                axes[1, 1].set_title('Right Foot Contact Force Components')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "foot_contact_forces.png"), dpi=150)
                plt.close()

                print(f"All plots saved to: {plot_dir}")
            elif len(time_data) > 0:
                print("Warning: matplotlib or numpy not installed, cannot plot figures")


def train(args):
    master_runner = prepare(args)
    if args.resume:
        master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
    master_runner.learn(max_iterations=args.iter)

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.eval:
        play(args)
    else:
        train(args)
