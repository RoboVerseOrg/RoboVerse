# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags

TAR_ACTOR_ID = 1

class HumanoidSpeed(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._speed_change_steps_min = cfg["env"]["speedChangeStepsMin"]
        self._speed_change_steps_max = cfg["env"]["speedChangeStepsMax"]

        self._add_input_noise = cfg["env"].get("addInputNoise", False)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._speed_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)

        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        reward_raw_num = 1
        if self.power_usage_reward:
            reward_raw_num += 1
        if self.power_reward:
            reward_raw_num += 1

        self.reward_raw = torch.zeros((self.num_envs, reward_raw_num)).to(self.device)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3

        if (self._add_input_noise):
            obs_size += 16
            
        if self.obs_v == 2:
            obs_size *= self.past_track_steps
        
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return
    
    def _update_marker(self):
        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2]
        self._marker_pos[..., 0] += 0.5 + 0.2 * self._tar_speed
        self._marker_pos[..., 2] = 0.0
        

        self._marker_rot[:] = 0
        self._marker_rot[:, -1] = 1.0
        

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "egoquest/data/assets/mjcf/"
        asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.max_angular_velocity = 0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)
        
        
        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + to_torch(self._marker_handles, device=self.device, dtype=torch.int32)
        
        
        return

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._speed_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        
        
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)
        
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._speed_change_steps_min, high=self._speed_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        
        self._tar_speed[env_ids] = tar_speed
        self._speed_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return
    
    def _compute_flip_task_obs(self, normal_task_obs, env_ids):
        B, D = normal_task_obs.shape
        flip_task_obs = normal_task_obs.clone()
        flip_task_obs[:, 1] = -flip_task_obs[:, 1]

        return flip_task_obs
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_speed = self._tar_speed
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_speed = self._tar_speed[env_ids]
        
        obs = compute_speed_observations(root_states, tar_speed)

        if self._add_input_noise:
            obs = torch.cat([obs, torch.randn((obs.shape[0], 16)).to(obs) * 0.1], dim=-1)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        
        # if False:
        if flags.test:
            root_pos = self._humanoid_root_states[..., 0:3]
            delta_root_pos = root_pos - self._prev_root_pos
            root_vel = delta_root_pos / self.dt
            tar_dir_speed = root_vel[..., 0]
            # print(self._tar_speed, tar_dir_speed)
        
        self.rew_buf[:] = self.reward_raw = compute_speed_reward(root_pos, self._prev_root_pos,  root_rot, self._tar_speed, self.dt)
        self.reward_raw = self.reward_raw[:, None]

        # if True:
        if self.power_reward:
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power = power_all.sum(dim=-1)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        # if True:
        if self.power_usage_reward: 
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power_all = power_all.reshape(-1, 23, 3)
            left_power = power_all[:, self.left_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            right_power = power_all[:, self.right_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            self.power_acc[:, 0] += left_power
            self.power_acc[:, 1] += right_power
            power_usage_reward = self.power_acc/(self.progress_buf + 1)[:, None]
            # print((power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs())
            power_usage_reward = - self.power_usage_coefficient * (power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
            power_usage_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.
            
            self.rew_buf[:] += power_usage_reward
            self.reward_raw = torch.cat([self.reward_raw, power_usage_reward[:, None]], dim=-1)
            

        return

    def _draw_task(self):
        self._update_marker()
        return
    
    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)
        self.power_acc[env_ids] = 0
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel  = super()._sample_ref_state(env_ids)
        
        # ZL Hack: Forcing to always be facing the x-direction. 
        heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_inv_repeat = heading_rot_inv[:, None].repeat(1, 24, 1)
        root_rot = quat_mul(heading_rot_inv, root_rot).clone()
        rb_pos = quat_apply(heading_rot_inv_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
        rb_rot = quat_mul(heading_rot_inv_repeat, rb_rot).clone()
        root_ang_vel = quat_apply(heading_rot_inv, root_ang_vel).clone()
        root_vel = quat_apply(heading_rot_inv, root_vel).clone()
        body_vel = quat_apply(heading_rot_inv_repeat, body_vel).clone()
        
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel 

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_speed')):
            self._output_motion_target_speed = []

        tar_speed = self._tar_speed[0].cpu().numpy()
        self._output_motion_target_speed.append(tar_speed)

        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_speed) > 1):
            output_data = np.array(self._output_motion_target_speed)
            np.save('output/record_tar_speed.npy', output_data)

            self._output_motion_target_speed = []

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_speed_observations(root_states, tar_speed):
    # type: (Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.zeros_like(root_states[..., 0:3])
    tar_dir3d[..., 0] = 1
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = torch_utils.my_quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)

    return obs

@torch.jit.script
def compute_speed_reward(root_pos, prev_root_pos, root_rot, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = root_vel[..., 0]
    tangent_speed = root_vel[..., 1]

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +  tangent_err_w * tangent_vel_err * tangent_vel_err))

    reward = dir_reward

    return reward