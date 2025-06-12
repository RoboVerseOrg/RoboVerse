"""Anymal locomotion task from IsaacGymEnvs.

This task trains a quadruped robot to follow velocity commands using RL.
"""

import torch
from typing import List, Optional

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.anymal_cfg import AnymalCfg as AnymalRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass
from metasim.utils.math import quat_rotate, quat_rotate_inverse

from ..base_task_cfg import BaseTaskCfg


@configclass
class AnymalCfg(BaseTaskCfg):
    """Configuration for Anymal locomotion task."""
    
    episode_length = 1000  # 20 seconds at 50Hz
    traj_filepath = None
    task_type = TaskType.LOCOMOTION
    
    # Normalization scales
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.5
    
    # Reward scales
    lin_vel_xy_reward_scale = 1.0
    ang_vel_z_reward_scale = 0.5
    torque_reward_scale = -0.000025
    
    # Command ranges
    command_x_range = [-2., 2.]  # m/s
    command_y_range = [-1., 1.]  # m/s  
    command_yaw_range = [-1., 1.]  # rad/s
    
    # Episode termination
    base_contact_force_threshold = 1.0
    knee_contact_force_threshold = 1.0
    
    # Robot configuration
    robot: AnymalRobotCfg = AnymalRobotCfg()
    
    # No objects needed for locomotion task
    objects: List[RigidObjCfg] = []
    
    observation_space = {
        "shape": [48]  # 48D observation
    }
    
    randomize = {
        "robot": {
            "anymal": {
                "joint_qpos": {
                    "type": "scaling",
                    "low": 0.5,
                    "high": 1.5,
                    "base": "default"  # Scale from default joint positions
                },
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                }
            }
        }
    }
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        super().__post_init__()
        self._commands = None
    
    def get_observation(self, states):
        """Get observations from states."""
        observations = []
        
        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["anymal"]
            
            # Get base orientation and velocities
            base_quat = torch.tensor(robot_state.get("rot", [1., 0., 0., 0.]), dtype=torch.float32)  # w,x,y,z
            base_lin_vel = torch.tensor(robot_state.get("lin_vel", robot_state.get("vel", [0., 0., 0.])), dtype=torch.float32)
            base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0., 0., 0.]), dtype=torch.float32)
            
            # Debug print
            if not hasattr(self, '_debug_printed'):
                print(f"Debug: robot_state keys: {robot_state.keys() if hasattr(robot_state, 'keys') else 'Not a dict'}")
                print(f"Debug: base_quat: {base_quat}")
                self._debug_printed = True
            
            # Transform velocities to base frame
            base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel) * self.lin_vel_scale
            base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel) * self.ang_vel_scale
            
            # Get gravity vector in base frame
            gravity_vec = torch.tensor([0., 0., -1.], dtype=torch.float32)
            projected_gravity = quat_rotate(base_quat, gravity_vec)
            
            # Get joint positions and velocities
            if "joint_qpos" in robot_state:
                dof_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32)
            elif "dof_pos" in robot_state:
                dof_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32)
            else:
                dof_pos = torch.zeros(12, dtype=torch.float32)
                
            # Get default positions for scaling
            default_pos = torch.tensor([
                0.03, 0.03, -0.03, -0.03,  # HAA
                0.4, -0.4, 0.4, -0.4,       # HFE
                -0.8, 0.8, -0.8, 0.8        # KFE
            ], dtype=torch.float32)
            
            dof_pos_scaled = (dof_pos - default_pos) * self.dof_pos_scale
            
            if "joint_qvel" in robot_state:
                dof_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32)
            elif "dof_vel" in robot_state:
                dof_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32)
            else:
                dof_vel = torch.zeros(12, dtype=torch.float32)
                
            dof_vel_scaled = dof_vel * self.dof_vel_scale
            
            # Get commands
            if self._commands is not None and i < len(self._commands):
                commands = self._commands[i]
            else:
                commands = torch.zeros(3, dtype=torch.float32)
                
            commands_scaled = commands * torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], dtype=torch.float32)
            
            # Get previous actions
            if hasattr(self, '_prev_actions') and self._prev_actions is not None and i < len(self._prev_actions):
                prev_actions = self._prev_actions[i]
            else:
                prev_actions = torch.zeros(12, dtype=torch.float32)
            
            # Build observation
            obs = torch.cat([
                base_lin_vel_base,     # 3
                base_ang_vel_base,     # 3
                projected_gravity,     # 3
                commands_scaled,       # 3
                dof_pos_scaled,        # 12
                dof_vel_scaled,        # 12
                prev_actions,          # 12
            ])  # Total: 48
            
            observations.append(obs)
            
        return torch.stack(observations) if observations else torch.zeros((0, 48))
    
    def reward_fn(self, states, actions):
        """Compute reward."""
        # Handle TensorState object from IsaacGym
        if hasattr(states, '__class__') and states.__class__.__name__ == 'TensorState':
            robot_state = states.robots["anymal"]
            
            # Get base state
            base_quat = robot_state.root_state[:, 3:7]
            base_lin_vel = robot_state.root_state[:, 7:10]
            base_ang_vel = robot_state.root_state[:, 10:13]
            
            # Transform to base frame
            base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
            base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)
            
            # Get commands
            if self._commands is not None:
                commands = self._commands
            else:
                num_envs = base_quat.shape[0]
                commands = torch.zeros((num_envs, 3), device=base_quat.device)
            
            # Velocity tracking rewards
            lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
            ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
            
            rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.lin_vel_xy_reward_scale
            rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.ang_vel_z_reward_scale
            
            # Torque penalty
            if hasattr(robot_state, 'torques'):
                torques = robot_state.torques
            else:
                torques = torch.zeros((base_quat.shape[0], 12), device=base_quat.device)
                
            rew_torque = torch.sum(torch.square(torques), dim=1) * self.torque_reward_scale
            
            # Total reward
            rewards = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
            rewards = torch.clamp(rewards, min=0.)
            
            return rewards
            
        else:
            # Handle list format
            rewards = []
            for i, env_state in enumerate(states):
                robot_state = env_state["robots"]["anymal"]
                
                # Get base state
                base_quat = torch.tensor(robot_state["rot"], dtype=torch.float32)
                base_lin_vel = torch.tensor(robot_state.get("lin_vel", robot_state.get("vel", [0., 0., 0.])), dtype=torch.float32)
                base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0., 0., 0.]), dtype=torch.float32)
                
                # Transform to base frame
                base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
                base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)
                
                # Get commands
                if self._commands is not None and i < len(self._commands):
                    commands = self._commands[i]
                else:
                    commands = torch.zeros(3, dtype=torch.float32)
                
                # Velocity tracking rewards
                lin_vel_error = torch.sum(torch.square(commands[:2] - base_lin_vel[:2]))
                ang_vel_error = torch.square(commands[2] - base_ang_vel[2])
                
                rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.lin_vel_xy_reward_scale
                rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.ang_vel_z_reward_scale
                
                # For now, skip torque penalty in list mode
                rew_torque = 0.0
                
                # Total reward
                reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
                reward = torch.clamp(reward, min=0.)
                
                rewards.append(reward.item())
                
            return torch.tensor(rewards) if rewards else torch.tensor([0.0])
    
    def termination_fn(self, states):
        """Check if episode should terminate."""
        # Handle TensorState object from IsaacGym
        if hasattr(states, '__class__') and states.__class__.__name__ == 'TensorState':
            robot_state = states.robots["anymal"]
            
            # Get contact forces
            if hasattr(robot_state, 'contact_forces'):
                contact_forces = robot_state.contact_forces
                
                # Check base contact
                base_contact = torch.norm(contact_forces[:, 0, :], dim=1) > self.base_contact_force_threshold
                
                # Check knee contacts (indices would need to be set properly)
                # For now, just check base
                terminations = base_contact
            else:
                # No contact forces available
                num_envs = robot_state.root_state.shape[0]
                terminations = torch.zeros(num_envs, dtype=torch.bool, device=robot_state.root_state.device)
                
            return terminations
            
        else:
            # Handle list format
            terminations = []
            for env_state in states:
                # For now, don't terminate based on contacts in list mode
                terminations.append(False)
                
            return torch.tensor(terminations) if terminations else torch.tensor([False])
    
    def build_scene(self, config=None):
        """Build the scene."""
        # Initialize commands
        self._commands = None
        self._prev_actions = None
        
    def reset(self, env_ids=None):
        """Reset task-specific state."""
        if env_ids is None:
            num_envs = 1
        else:
            num_envs = len(env_ids)
            
        # Generate random velocity commands
        commands = torch.zeros((num_envs, 3), dtype=torch.float32)
        commands[:, 0] = torch.rand(num_envs) * (self.command_x_range[1] - self.command_x_range[0]) + self.command_x_range[0]
        commands[:, 1] = torch.rand(num_envs) * (self.command_y_range[1] - self.command_y_range[0]) + self.command_y_range[0]
        commands[:, 2] = torch.rand(num_envs) * (self.command_yaw_range[1] - self.command_yaw_range[0]) + self.command_yaw_range[0]
        
        if self._commands is None:
            self._commands = commands
        else:
            if env_ids is not None:
                self._commands[env_ids] = commands
            else:
                self._commands = commands
                
        # Reset previous actions
        if self._prev_actions is None:
            self._prev_actions = torch.zeros((num_envs, 12), dtype=torch.float32)
        else:
            if env_ids is not None:
                self._prev_actions[env_ids] = 0.0
                
    def post_reset(self):
        """Called after reset."""
        pass