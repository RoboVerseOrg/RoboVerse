# h2o_wrapper.py
"""
Minimal rsl_rl-compatible wrapper for the H2O legged robot.
Fill in the TODO blocks as you port features.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List

import torch

from metasim.cfg.scenario import ScenarioCfg
from phc.utils import torch_utils
# -- project specific (adapt paths/names) ---------------------------------
from metasim.cfg.tasks.h2o.base_legged_cfg import BaseLeggedTaskCfg
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper
from roboverse_learn.skillblender_rl.utils import (
    get_body_reindexed_indices_from_substring,
    get_joint_reindexed_indices_from_substring,
)

# -------------------------------------------------------------------------


class H2OWrapper(RslRlWrapper):
    """rsl_rl vector-env wrapper for H2O."""

    # ------------------------------------------------------------------ #
    # 1. ctor & indices                                                  #
    # ------------------------------------------------------------------ #
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.up_axis_idx = 2                             # z-up world

        # ---- static indices (used by rewards / observations) ----------
        self._parse_rigid_body_indices(scenario.robots[0])
        self._parse_joint_indices(scenario.robots[0])

        # ---- cfg-level meta -------------------------------------------
        self.dt = scenario.decimation * scenario.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim
        self._prepare_reward_function(scenario.task)
        self._init_buffers()

    # ---------------- rigid-body indices ------------------------------ #
    def _parse_rigid_body_indices(self, robot_cfg):
        """Resolve and cache body indices once at start-up."""
        self.feet_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot_cfg.name, robot_cfg.feet_links, device=self.device
        )
        self.cfg.feet_indices = self.feet_indices

    # ---------------- joint indices ----------------------------------- #
    def _parse_joint_indices(self, robot_cfg):
        """Resolve joint groups (only if needed by reward/obs)."""
        self.cfg.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env.handler, robot_cfg.name, robot_cfg.upper_body_joints, device=self.device
        )

    # ------------------------------------------------------------------ #
    # 2. runtime buffers                                                 #
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        n, obs_dim, act_dim = self.num_envs, self.num_obs, self.num_actions

        self.obs_buf   = torch.zeros(n, obs_dim, device=self.device)
        self.rew_buf   = torch.zeros(n, device=self.device)
        self.reset_buf = torch.ones (n, dtype=torch.bool, device=self.device)

        self.actions   = torch.zeros(n, act_dim, device=self.device)
        self.base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(n, 1)
        self.gravity_vec = torch.tensor([0, 0, -1], device=self.device).repeat(n, 1)

        # History stacks for frame-stacked observations
        self.obs_history = deque(
            [torch.zeros_like(self.obs_buf)] * self.cfg.frame_stack,
            maxlen=self.cfg.frame_stack,
        )

    # ------------------------------------------------------------------ #
    # 3. core RL hooks                                                   #
    # ------------------------------------------------------------------ #
    def clip_actions(self, a: torch.Tensor) -> torch.Tensor:
        lim = self.cfg.normalization.clip_actions
        return torch.clamp(a, -lim, lim)

    def _pre_physics_step(self, a: torch.Tensor) -> torch.Tensor:
        a = self.clip_actions(a)
        self.actions[:] = a
        return a

    def _physics_step(self, actions: torch.Tensor):
        """
        Isaac/MetaSim env.step – must return terminated / timeout flags.
        Only terminated|timeout are required here.
        """
        env_state, _, term, tout, _ = self.env.step(actions)
        self.reset_buf = term | tout
        return env_state

    def _post_physics_step(self, env_state):
        # --- reward (simple placeholder) ------------------------------
        self.rew_buf[:] = 0.0   # TODO: plug actual reward

        # --- observation ---------------------------------------------
        self._compute_observations(env_state)

        # --- book-keeping --------------------------------------------
        self.obs_history.append(self.obs_buf.clone())

        return self.obs_buf, None, self.rew_buf

    # public API
    def step(self, actions):
        acts = self._pre_physics_step(actions)
        st   = self._physics_step(acts)
        obs, priv, rew = self._post_physics_step(st)
        return obs, priv, rew, self.reset_buf, {}

    def reset(self, env_ids: List[int] | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if not env_ids:
            return
        self.env.reset(self.init_states, env_ids)
        self.reset_buf[env_ids] = False
        # clear history
        for h in self.obs_history:
            h[env_ids] = 0

    # ------------------------------------------------------------------ #
    # 4. observation assembly                                            #
    # ------------------------------------------------------------------ #
    def _compute_observations(self, env_state):
        """
        Build actor observation.
        Example below uses only DOF pos & vel;
        Expand with reference-tracking or vision if you need.
        """
        dof_pos = env_state.robots[self.robot.name].dof_pos
        dof_vel = env_state.robots[self.robot.name].dof_vel
        proj_g  = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.obs_buf[:] = torch.cat([dof_pos, dof_vel, proj_g], dim=-1)

    # ------------------------------------------------------------------ #
    # 5. reward functions (skeleton)                                     #
    # ------------------------------------------------------------------ #
    def _prepare_reward_function(self, task: BaseLeggedTaskCfg):
        """Register reward fns according to cfg.scales dict."""
        self.reward_scales: dict[str, float] = task.reward_weights
        self.reward_fns: dict[str, Callable] = {}      # name → callable
        for name, scale in self.reward_scales.items():
            fn_name = f"reward_{name}"
            if hasattr(self, fn_name):
                self.reward_fns[name] = getattr(self, fn_name)
            else:
                print(f"[H2OWrapper] WARNING: reward fn {fn_name} missing")

    # Example reward stub
    def reward_alive(self, env_state, robot, cfg):
        return torch.ones(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # 6. optional curriculum / pushes (leave blank if not needed)        #
    # ------------------------------------------------------------------ #
    def _post_physics_step_callback(self):  # called each sim-step
        pass

    def _push_robots(self):                 # random impulses
        pass

    def begin_seq_motion_samples(self):
        pass

    def forward_motion_samples(self):
        pass

    def compute_self_and_task_obs(self, ):
        """ Computes observations
        """
        #import pdb;pdb.set_trace()
        #import ipdb; ipdb.set_trace()
        # print("self.episode_length_buf: ", self.episode_length_buf)
        offset = self.env_origins + self.env_origins_init_3Doffset
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
    

        ref_body_pos = motion_res["rg_pos"] 
        ref_body_pos_extend = motion_res["rg_pos_t"]
        ref_body_vel_subset = motion_res["body_vel"] # [num_envs, num_markers, 3]
        ref_body_vel = ref_body_vel_subset
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
        ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]

        
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)
            
        if self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-full':




            #TODO change into roboverse state
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self.dof_pos
            dof_vel = self.dof_vel

                    # robot
            dof_pos = self.dof_pos
            dof_vel = self.dof_vel
            base_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            base_gravity = self.projected_gravity



            extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
            body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

            body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
            body_pos_subset_student = body_pos_extend[:, self._track_bodies_extend_id[-3:], :] #-3 means student obs

            extend_curr_rot = body_rot[:, self.extend_body_parent_ids].clone()
            body_rot_extend = torch.cat([body_rot, extend_curr_rot], dim=1)
            body_rot_subset = body_rot_extend[:, self._track_bodies_extend_id, :]

            body_vel_extend = torch.cat([body_vel, body_vel[:, self.extend_body_parent_ids].clone()], dim=1)
            body_vel_subset = body_vel_extend[:, self._track_bodies_extend_id, :]

            body_ang_vel_extend = torch.cat([body_ang_vel, body_ang_vel[:, self.extend_body_parent_ids].clone()], dim=1)
            body_ang_vel_subset = body_ang_vel_extend[:, self._track_bodies_extend_id, :]

        
            ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
            ref_rb_pos_subset_student = ref_body_pos_extend[:, self._track_bodies_extend_id[-3:]]
            ref_rb_rot_subset = ref_body_rot_extend[:, self._track_bodies_extend_id]
            ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
            ref_body_vel_subset_student = ref_body_vel_extend[:, self._track_bodies_extend_id[-3:]]
            ref_body_ang_vel_subset = ref_body_ang_vel_extend[:, self._track_bodies_extend_id]
            

            # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
            root_pos = body_pos[..., 0, :]
            root_rot = body_rot[..., 0, :]
            root_vel = body_vel[:, 0, :]
            root_ang_vel = body_ang_vel[:, 0, :]
            ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
  
            # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
            # import ipdb; ipdb.set_trace()
            if self.cfg.motion.realtime_vr_keypoints:
                ref_rb_pos_subset = self.realtime_vr_keypoints_pos
                ref_body_vel_subset = self.realtime_vr_keypoints_vel
                assert self.cfg.motion.num_traj_samples == 1

            if self.cfg.asset.clip_motion_goal:
                # import ipdb; ipdb.set_trace()
                ref_head = ref_rb_pos_subset_student[:, 2]
                body_xyz = self.root_states[:, :3]
                direction_to_body = body_xyz - ref_head
                xy_direction = direction_to_body[:,:2]
                distance = torch.norm(xy_direction, dim=1)
                # import ipdb; ipdb.set_trace()
                far = distance > self.cfg.asset.clip_motion_goal_distance
                direction_to_body_norm = F.normalize(direction_to_body[:,:2], p = 2, dim=1)
                # direction_to_body_norm = xy_direction / 
                ref_rb_pos_subset_student[far, 2, :2] = self.root_states[far, :2] - direction_to_body_norm[far] * self.cfg.asset.clip_motion_goal_distance

            task_obs = self.compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset_student, ref_rb_pos_subset_student, ref_body_vel_subset_student,  self.cfg.motion.num_traj_samples , ref_episodic_offset = self.ref_episodic_offset)
            task_obs_full = self.compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_extend, ref_body_pos_extend, ref_body_vel_extend,  self.cfg.motion.num_traj_samples , ref_episodic_offset = self.ref_episodic_offset,obs_full = self.cfg.obs_full)

            if self.cfg.obs_full:
                obs = torch.cat([dof_pos, #19 dim
                        dof_vel, #19 dim
                        base_ang_vel, #3 dim
                        base_gravity, #3 dim 
                        task_obs_full, #207 dim
                        ],dim=-1)
            else:
                obs = torch.cat([dof_pos, #19 dim
                                        dof_vel, #19 dim
                                        base_ang_vel, #3 dim
                                        base_gravity, #3 dim 
                                        task_obs, # 27 dim ; 71 dim in total
                                        ],dim=-1)
            if self.cfg.noise.add_noise:
                if self.cfg.obs_full:
                        noise_vec = torch.zeros_like(obs[0])
                        noise_scales = self.cfg.noise.noise_scales
                        noise_level = self.cfg.noise.noise_level
                        noise_vec[0:self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                        noise_vec[self.num_dof :2*self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                        noise_vec[2*self.num_dof   : 2*self.num_dof + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                        noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.gravity * noise_level
                        #import ipdb;ipdb.set_trace()
                        noise_vec[2*self.num_dof + 6 : 2*self.num_dof + 6 + 207] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos
                else:
                        noise_vec = torch.zeros_like(obs[0])
                        noise_scales = self.cfg.noise.noise_scales
                        noise_level = self.cfg.noise.noise_level
                        noise_vec[0:self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                        noise_vec[self.num_dof :2*self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                        noise_vec[2*self.num_dof   : 2*self.num_dof + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                        noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.gravity * noise_level
                        noise_vec[2*self.num_dof + 6 : 2*self.num_dof + 6 + 27] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos
                        #print("Noise scale: ",noise_vec )
                obs += (2 * torch.rand_like(obs) - 1) * noise_vec 

        else:
            raise NotImplementedError

        return obs 
    
    def _get_state_from_motionlib_cache_trimesh(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        #import ipdb;ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        #import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        # self.root_states[:,:2] = motion_res['root_pos'][:, :2]

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache
    

    def compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos,   ref_body_pos, ref_body_vel, time_steps,  ref_episodic_offset = None, ref_vel_in_task_obs = True, obs_full = False ):    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,  int, bool, bool) -> Tensor
        #  Teleop version
        obs = []
        B, J, _ = body_pos.shape

        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
        heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
        
        ##### Body position and rotation differences
        diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
        #import ipdb;ipdb.set_trace()
        diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)) # 
        #import ipdb;ipdb.set_trace()
        ##### body pos + Dof_pos This part will have proper futuers.
        local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
        local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
        
        local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

        if ref_episodic_offset is not None:
            # import ipdb; ipdb.set_trace()
            diff_global_body_pos_offset= ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, J, -1)
            # diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset.view(-1, 3)
            diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset
            local_ref_body_pos_offset = ref_episodic_offset.repeat(J,1)[:J * ref_episodic_offset.shape[0], :]
            if obs_full :
                #import ipdb;ipdb.set_trace()
                local_ref_body_pos[0::1] += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)[0::1]
            else:
                #import ipdb;ipdb.set_trace()
                local_ref_body_pos[2::3] += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)[2::3]
            # local_ref_body_pos += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)

        # make some changes to how futures are appended.
        #import ipdb;ipdb.set_trace()
        obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
        obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
        if ref_vel_in_task_obs:
            obs.append(local_ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3

        obs = torch.cat(obs, dim=-1).view(B, -1)
        
        return obs


    @property
    def _motion_lib(self):      
        return self._motion_lib
    @property
    def default_dof_pos(self):  
        return self.default_dof_pos
    @property
    def dof_pos_limits(self):   
        return self.dof_pos_limits

