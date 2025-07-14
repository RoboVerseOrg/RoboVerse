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

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        # create some wrapper tensors for different slices
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        # if (not self.headless) and self.cfg.motion.visualize and self.cfg.motion.teleop:
        #     self.root_states = self.root_states_all[0::(self._num_teleop_markers+1)]
        #     self.marker_states = [self.root_states_all[marker_i::(self._num_teleop_markers + 1)] for marker_i in range(1,self._num_teleop_markers + 1)]
        #     # self.root_states = self.root_states_all[0::(self.cfg.motion.num_markers+1)]
        #     # self.marker_states = [self.root_states_all[marker_i::(self.cfg.motion.num_markers + 1)] for marker_i in range(1,self.cfg.motion.num_markers + 1)]
        # else:
        #     self.root_states = self.root_states_all

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        #import ipdb;ipdb.set_trace()
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # init rigid body state
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        # self._rigid_body_pos = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 0:3]
        # self._rigid_body_rot = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 3:7]
        # self._rigid_body_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 7:10]
        # self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 10:13]

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_root_pos = torch.zeros_like(self.root_states[:, 0:3])
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # normalization
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)


        self.measured_heights = 0

        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel) # different from self.last_root_vel


        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Init for motion reference
        if self.cfg.motion.teleop:
            self.ref_motion_cache = {}
            self._load_motion()
            self.marker_coords = torch.zeros(self.num_envs, (self.num_dofs + (4 if self.cfg.motion.extend_head else 3)) * self.cfg.motion.num_traj_samples, 3, dtype=torch.float, device=self.device, requires_grad=False) # extend
            self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
            self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
            self.motion_ids = torch.arange(self.num_envs).to(self.device)
            self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            self.base_pos_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            if self.cfg.motion.teleop:
                self._recovery_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                self._package_loss_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

            self.ref_base_pos_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_rot_init = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_vel_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_ang_vel_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            self.env_origins_init_3Doffset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self._resample_motion_times(env_ids) #need to resample before reset root states
            # self._update_motion_reference()
            self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)



    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.



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

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.last_episode_length_buf = self.episode_length_buf.clone()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.cfg.motion.teleop:
            self._update_recovery_count()

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

        self.base_ang_vel[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self._rigid_body_ang_vel[:, 11, :])


        self.projected_gravity[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.gravity_vec)

        self._post_physics_step_callback()
        # compute observations, rewards, resets, ...
        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0 and self.cfg.play_in_order:
            self.motion_idxes += 1
        self.reset_idx(env_ids)
        #print("obs_buf_before",self.obs_buf)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        #print("obs_buf_after",self.obs_buf)
        #import ipdb;ipdb.set_trace()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, 0:3]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if self.cfg.env.im_eval:
            offset = self.env_origins + self.env_origins_init_3Doffset
            time = (self.episode_length_buf) * self.dt + self.motion_start_times
            # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, time, offset)
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, time, offset= offset)

            ref_body_pos_extend = motion_res['rg_pos_t']

            body_rot = self._rigid_body_rot
            body_pos = self._rigid_body_pos

            extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
            body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

            diff_global_body_pos = ref_body_pos_extend - body_pos_extend

            self.extras['mpjpe'] = (diff_global_body_pos).norm(dim=-1).mean(dim=-1)
            self.extras['body_pos'] = body_pos_extend.cpu().numpy()
            self.extras['body_pos_gt'] = ref_body_pos_extend.cpu().numpy()


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

    def _compute_observations(self, env_state):
        self.obs_buf = self.compute_self_and_task_obs()
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
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

        # if self.cfg.motion.teleop:
            # self.motion_times += self.dt # TODO: align with motion_dt. ZL: don't need that, motion lib will handle it.
            # self._update_motion_reference()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.motion.teleop and (self.common_step_counter % self.cfg.motion.resample_motions_for_envs_interval == 0):
            if self.cfg.motion.resample_motions_for_envs:
                print("Resampling motions for envs")
                print("common_step_counter: ", self.common_step_counter)
                self.resample_motion()


    def _push_robots(self):                 # random impulses
        pass

    def begin_seq_motion_samples(self):
        pass

    def forward_motion_samples(self):
        pass

    def compute_self_and_task_obs(self):
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
