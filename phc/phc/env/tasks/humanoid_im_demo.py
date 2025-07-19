
from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im as humanoid_im
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL 

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from collections import defaultdict
import aiohttp, cv2, asyncio, json


class HumanoidImDemo(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.j3d = torch.zeros([1, 24, 3]).to(self.device).float()
        self.j3d_vel = torch.zeros([1, 24, 3]).to(self.device).float()

    async def talk(self):
        URL = 'http://0.0.0.0:8081/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            self.ws = ws
            await ws.send_str("get_pose")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close cmd':
                        await ws.close()
                        break
                    else:
                        json_data = json.loads(msg.data)
                        self.j3d = torch.tensor(json_data["j3d_curr"]).to(self.device).float()
                        self.j3d_vel = torch.tensor(json_data["j3d_curr_vel"]).to(self.device).float()

                        await ws.send_str("get_pose")

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    def _update_marker(self):
        if flags.show_traj:
            self._marker_pos[:] = 0
        else:
            self._marker_pos[:] = self.ref_body_pos

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = torch_utils.calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))

        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._sample_ref_state(env_ids)

        from scipy.spatial.transform import Rotation as sRot
        random_heading_quat = torch.from_numpy(sRot.from_euler("xyz", [0, 0, np.pi]).as_quat())[None,].float().to(self.device)
        random_heading_quat_repeat = random_heading_quat[:, None].repeat(1, 24, 1)
        root_rot = quat_mul(random_heading_quat, root_rot).clone()
        rb_pos = quat_apply(random_heading_quat_repeat, rb_pos - root_pos[:, None, :]).clone()
        rb_rot = quat_mul(random_heading_quat_repeat, rb_rot).clone()
        root_ang_vel = quat_apply(random_heading_quat, root_ang_vel).clone()
        rb_pos = rb_pos + (self.j3d[0, 0:1, :] - root_pos)
        root_pos = self.j3d[0, 0:1, :]
        root_pos[..., 2] = 0.93

        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel, rigid_body_pos=rb_pos, rigid_body_rot=rb_rot, rigid_body_vel=body_vel, rigid_body_ang_vel=body_ang_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times
        self._sampled_motion_ids[env_ids] = motion_ids
        if flags.follow:
            self.start = True  ## Updating camera when reset
        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs_demo(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs

        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:10].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, 5))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs
        return obs

    def _compute_task_obs_demo(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]

        # ref_rb_pos = self.j3d[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d.shape[0]]
        # ref_body_vel = self.j3d_vel[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d_vel.shape[0]]
        ref_rb_pos = self.j3d
        ref_body_vel = self.j3d_vel
        time_steps = 1

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]

        if self.zero_out_far:
            close_distance = 0.25
            distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

            zeros_subset = distance > close_distance
            ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
            ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]

        obs = humanoid_im.compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)

        if len(env_ids) == self.num_envs:
            self.ref_body_pos = ref_rb_pos
            self.ref_body_pos_subset = ref_rb_pos_subset
            self.ref_pose_aa = None

        return obs
