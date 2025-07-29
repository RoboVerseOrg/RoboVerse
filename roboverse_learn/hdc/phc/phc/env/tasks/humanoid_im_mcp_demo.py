import json
import subprocess
from collections import deque

import aiohttp
import numpy as np
import requests
import scipy.ndimage.filters as filters
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
from uhc.utils.transform_utils import quat_correct_two_batch

import phc.env.tasks.humanoid_im as humanoid_im
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp
import phc.utils.pytorch3d_transforms as ptr
from phc.utils.flags import flags

SERVER = "*.*.*.*"
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]


class HumanoidImMCPDemo(humanoid_im_mcp.HumanoidImMCP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        ## Debugging
        # self.res_data = joblib.load("/home/zhengyiluo5/dev/meta/HybrIK/ik_res.pkl")
        # self.rot_mat_ref = torch.from_numpy(sRot.from_rotvec(np.array(self.res_data['pose_aa']).reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)).float().to(self.device)
        ## Debugging

        self.local_translation_batch = self.skeleton_trees[0].local_translation[None,]
        self.parent_indices = self.skeleton_trees[0].parent_indices
        self.pose_mat = torch.eye(3).repeat(self.num_envs, 24, 1, 1).to(self.device)
        self.trans = torch.zeros(self.num_envs, 3).to(self.device)

        self.prev_ref_body_pos = torch.zeros(self.num_envs, 24, 3).to(self.device)
        self.prev_ref_body_rot = torch.zeros(self.num_envs, 24, 4).to(self.device)

        self.zero_trans = torch.zeros([self.num_envs, 3])
        self.s_dt = 1 / 30

        self.to_isaac_mat = torch.from_numpy(
            sRot.from_euler("xyz", np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        ).float()
        self.to_global = torch.from_numpy(sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()).float()

        self.root_pos_acc = deque(maxlen=30)
        self.body_rot_acc = deque(maxlen=30)
        self.body_pos_acc = deque(maxlen=30)

        flags.no_collision_check = True
        flags.show_traj = True
        self.close_distance = 0.5
        self.mean_limb_lengths = np.array([0.1061, 0.3624, 0.4015, 0.1384, 0.1132], dtype=np.float32)[None, :]

    async def talk(self):
        URL = f"http://{SERVER}:8080/ws"
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == "close cmd":
                        await ws.close()
                        break
                    else:
                        print(msg.data)
                        try:
                            msg = json.loads(msg.data)
                            if msg["action"] == "reset":
                                self.reset()
                            elif msg["action"] == "start_record":
                                subprocess.Popen(["simplescreenrecorder", "--start-recording"])
                                print("start recording!!!!")
                                # self.recording = True
                            elif msg["action"] == "end_record":
                                print("end_recording!!!!")
                                if not self.recording:
                                    print("Not recording")
                                else:
                                    self.recording = False
                                    self.recording_state_change = True
                            elif msg["action"] == "set_env":
                                query = msg["query"]
                                env_id = query["env"]
                                self.viewing_env_idx = int(env_id)
                                print("view env idx: ", self.viewing_env_idx)
                        except:
                            import ipdb

                            ipdb.set_trace()
                            print("error parsing server message")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    def _update_marker(self):
        if flags.show_traj:
            self._marker_pos[:] = self.ref_body_pos
        else:
            self._marker_pos[:] = 0

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = torch_utils.calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if self._enable_task_obs:
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
        if env_ids is None:
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
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        if self.obs_v == 6:
            raise NotImplementedError
            # This part is not as good. use obs_v == 7 instead.
            # ref_rb_pos = self.j3d[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d.shape[0]]
            # ref_body_vel = self.j3d_vel[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d_vel.shape[0]]
            # pose_mat = self.pose_mat.clone()
            # trans = self.trans.clone()

            # pose_mat = self.rot_mat_ref[((self.progress_buf[env_ids] + 1) / 2).long() % self.rot_mat_ref.shape[0]] # debugging
            pose_res = requests.get(f"http://{SERVER}:8080/get_pose")
            json_data = pose_res.json()
            pose_mat = torch.tensor(json_data["pose_mat"])[None,].float()
            # trans = torch.tensor(json_data["trans"]).to(self.device).float()

            trans = np.array(json_data["trans"]).squeeze()
            s_dt = json_data["dt"]
            self.root_pos_acc.append(trans)
            filtered_trans = filters.gaussian_filter1d(self.root_pos_acc, 3, axis=0, mode="mirror")
            trans = torch.tensor(filtered_trans[-1]).float()

            new_root = self.to_isaac_mat.matmul(pose_mat[:, 0])
            pose_mat[:, 0] = new_root
            trans = trans.matmul(self.to_isaac_mat.T)
            _, global_rotation = humanoid_kin.forward_kinematics_batch(
                pose_mat[:, smpl_2_mujoco], self.zero_trans, self.local_translation_batch, self.parent_indices
            )

            ref_rb_rot = ptr.matrix_to_quaternion_ijkr(global_rotation.matmul(self.to_global))

            ##################  ##################
            ref_rb_rot_np = ref_rb_rot.numpy()[0]

            if len(self.body_rot_acc) > 0:
                ref_rb_rot_np = quat_correct_two_batch(self.body_rot_acc[-1], ref_rb_rot_np)
                filtered_quats = filters.gaussian_filter1d(
                    np.concatenate([self.body_rot_acc, ref_rb_rot_np[None,]], axis=0), 1, axis=0, mode="mirror"
                )
                new_quat = filtered_quats[-1] / np.linalg.norm(filtered_quats[-1], axis=1)[:, None]
                self.body_rot_acc.append(new_quat)  # add the filtered quat.

                # pose_quat_global = np.array(self.body_rot_acc)
                # select_quats = np.linalg.norm(pose_quat_global[:-1, :] - pose_quat_global[1:, :], axis=2) > np.linalg.norm(pose_quat_global[:-1, :] + pose_quat_global[1:, :], axis=2)
                ref_rb_rot = torch.tensor(new_quat[None,]).float()
            else:
                self.body_rot_acc.append(ref_rb_rot_np)

            ################## ##################

            ref_rb_pos = SkeletonState.from_rotation_and_root_translation(
                self.skeleton_trees[0], ref_rb_rot, trans, is_local=False
            ).global_translation.to(self.device)  # SLOWWWWWWW
            ref_rb_rot = ref_rb_rot.to(self.device)
            ref_rb_pos = ref_rb_pos.to(self.device)
            ref_body_ang_vel = SkeletonMotion._compute_angular_velocity(
                torch.stack([self.prev_ref_body_rot, ref_rb_rot], dim=1), time_delta=s_dt, guassian_filter=False
            )[:, 0]
            ref_body_vel = SkeletonMotion._compute_velocity(
                torch.stack([self.prev_ref_body_pos, ref_rb_pos], dim=1), time_delta=s_dt, guassian_filter=False
            )[:, 0]  # this is slow!

            time_steps = 1
            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_rb_rot_subset[zeros_subset, 1:] = body_rot_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                ref_body_ang_vel_subset[zeros_subset, :] = body_ang_vel_subset[zeros_subset, :]

                far_distance = 3  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = (
                    (ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0])
                    / distance[vector_zero_subset, None]
                    * far_distance
                ) + body_pos_subset[vector_zero_subset, 0]

            obs = humanoid_im.compute_imitation_observations_v6(
                root_pos,
                root_rot,
                body_pos_subset,
                body_rot_subset,
                body_vel_subset,
                body_ang_vel_subset,
                ref_rb_pos_subset,
                ref_rb_rot_subset,
                ref_body_vel_subset,
                ref_body_ang_vel_subset,
                time_steps,
                self._has_upright_start,
            )

            self.prev_ref_body_pos = ref_rb_pos
            self.prev_ref_body_rot = ref_rb_rot
        elif self.obs_v == 7:
            pose_res = requests.get(f"http://{SERVER}:8080/get_pose")
            json_data = pose_res.json()
            ref_rb_pos = np.array(json_data["j3d"])[: self.num_envs, smpl_2_mujoco]
            trans = ref_rb_pos[:, [0]]

            # if len(self.root_pos_acc) > 0 and np.linalg.norm(trans - self.root_pos_acc[-1]) > 1:
            # import ipdb; ipdb.set_trace()
            # print("juping!!")
            ref_rb_pos_orig = ref_rb_pos.copy()

            ref_rb_pos = ref_rb_pos - trans
            ############################## Limb Length ##############################
            limb_lengths = []
            for i in range(6):
                parent = self.skeleton_trees[0].parent_indices[i]
                if parent != -1:
                    limb_lengths.append(np.linalg.norm(ref_rb_pos[:, parent] - ref_rb_pos[:, i], axis=-1))
            limb_lengths = np.array(limb_lengths).transpose(1, 0)
            scale = (limb_lengths / self.mean_limb_lengths).mean(axis=-1)
            ref_rb_pos /= scale[:, None, None]
            ############################## Limb Length ##############################
            s_dt = 1 / 30

            self.root_pos_acc.append(trans)
            filtered_root_trans = np.array(self.root_pos_acc)
            filtered_root_trans[..., 2] = filters.gaussian_filter1d(
                filtered_root_trans[..., 2], 10, axis=0, mode="mirror"
            )  # More filtering on the root translation
            filtered_root_trans[..., :2] = filters.gaussian_filter1d(
                filtered_root_trans[..., :2], 5, axis=0, mode="mirror"
            )
            trans = filtered_root_trans[-1]

            self.body_pos_acc.append(ref_rb_pos)
            body_pos = np.array(self.body_pos_acc)
            filtered_ref_rb_pos = filters.gaussian_filter1d(body_pos, 2, axis=0, mode="mirror")
            ref_rb_pos = filtered_ref_rb_pos[-1]

            ref_rb_pos = torch.from_numpy(ref_rb_pos + trans).float()
            ref_rb_pos = ref_rb_pos.matmul(self.to_isaac_mat.T).cuda()

            ref_body_vel = SkeletonMotion._compute_velocity(
                torch.stack([self.prev_ref_body_pos, ref_rb_pos], dim=1), time_delta=s_dt, guassian_filter=False
            )[:, 0]  #

            time_steps = 1
            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = (
                    (ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0])
                    / distance[vector_zero_subset, None]
                    * far_distance
                ) + body_pos_subset[vector_zero_subset, 0]

            obs = humanoid_im.compute_imitation_observations_v7(
                root_pos,
                root_rot,
                body_pos_subset,
                body_vel_subset,
                ref_rb_pos_subset,
                ref_body_vel_subset,
                time_steps,
                self._has_upright_start,
            )

            self.prev_ref_body_pos = ref_rb_pos

        if len(env_ids) == self.num_envs:
            self.ref_body_pos = ref_rb_pos
            self.ref_body_pos_subset = torch.from_numpy(ref_rb_pos_orig)
            self.ref_pose_aa = None

        return obs

    def _compute_reset(self):
        self.reset_buf[:] = 0
        self._terminate_buf[:] = 0
