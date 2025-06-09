from metasim.cfg.policy.base_policy import BasePolicyCfg

try:
    from curobo.types.math import Pose
    from pytorch3d import transforms

    from metasim.utils.kinematics_utils import get_curobo_models
except ImportError:
    pass

import torch
from diffusion_policy.common.pytorch_util import dict_apply
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg


class PolicyRunner:
    """Base class to run a policy, based on the policyCFG it preprocesses the observation to
    match the policy's input requirements and postprocesses the action into joint space to match the policy's action type
    """

    def __init__(self, scenario: ScenarioCfg, num_envs: int = 1, **kwargs):
        self.num_envs = num_envs
        self.scenario = scenario
        self.policy_cfg: BasePolicyCfg = None
        self.step = 0
        self.device = kwargs.get("device", "cuda:0")
        self.task_name = kwargs.get("task_name")
        self._init_policy(**kwargs)
        self.robot_ik = None
        self.curobo_n_dof = None
        self.action_cache = []
        self.__post_init__()

    def _init_policy(self, **kwargs):
        """Method for subclasses to inherit to load in the policy"""
        raise NotImplementedError

    def __post_init__(self):
        if self.policy_cfg.action_config.action_type == "ee":
            *_, self.robot_ik = get_curobo_models(self.scenario.robots[0])
            self.curobo_n_dof = len(self.robot_ik.robot_config.cspace.joint_names)
            self.ee_n_dof = len(self.scenario.robots[0].gripper_open_q)

        if self.policy_cfg.action_config.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.num_envs,
                    self.scenario.episode_length,
                    self.scenario.episode_length + self.policy_cfg.action_config.action_chunk_steps,
                    self.policy_cfg.action_config.action_dim,
                ],
                device=self.device,
            )
            self.k = 0.01

    def process_obs(self, obs):
        """
        Processes the observation to be used by the policy, according to the observation the policy is configured to use.
        """
        obs = dict_apply(obs, lambda x: x.to(device=self.device) if isinstance(x, torch.Tensor) else x)
        obs_dict = {}

        if self.policy_cfg.obs_config.norm_image:
            obs_dict["head_cam"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
        else:
            obs_dict["head_cam"] = obs["rgb"]
        if self.policy_cfg.obs_config.obs_type == "joint_pos":
            obs_dict["agent_pos"] = obs["joint_qpos"]
        if self.policy_cfg.obs_config.obs_type == "ee":
            robot_ee_state = obs["robot_ee_state"]
            robot_root_state = obs["robot_root_state"]
            robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = transforms.quaternion_apply(
                transforms.quaternion_invert(robot_quat), curr_ee_pos - robot_pos
            )
            curr_ee_quat_local = transforms.quaternion_multiply(transforms.quaternion_invert(robot_quat), curr_ee_quat)

            if self.policy_cfg.obs_config.ee_cfg.gripper_rep == "q_pos":
                gripper_state = obs["joint_qpos"][:, -2:]
            else:
                gripper_state = obs["robot_ee_state"][:, -1]

            if self.policy_cfg.obs_config.ee_cfg.rotation_rep == "quaternion":
                curr_ee_rot_local = curr_ee_quat_local
            else:
                curr_ee_rot_local = transforms.matrix_to_euler_angles(
                    transforms.quaternion_to_matrix(curr_ee_quat_local), convention="XYZ"
                )

            obs_dict["agent_pos"] = torch.cat([curr_ee_pos_local, curr_ee_rot_local, gripper_state], dim=1)

        if self.policy_cfg.obs_config.obs_padding > 0:
            padding_len = self.policy_cfg.obs_config.obs_padding - obs_dict["agent_pos"].shape[1]
            padding = torch.zeros(self.num_envs, padding_len, device=self.device)
            obs_dict["agent_pos"] = torch.cat([obs_dict["agent_pos"], padding], dim=1)

        assert obs_dict["agent_pos"].shape == (self.num_envs, self.policy_cfg.obs_config.obs_dim)
        # flush unused keys
        obs_dict = {k: v for k, v in obs_dict.items() if k in self.policy_cfg.obs_config.obs_keys}
        return obs_dict

    def action_to_dict(self, curr_action):
        """
        Converts action tensor to dict with joint keys
        """
        actions = [
            {
                "dof_pos_target": {
                    joint_name: curr_action[i, index]
                    for index, joint_name in enumerate(sorted(self.scenario.robots[0].joint_limits.keys()))
                }
            }
            for i in range(self.num_envs)
        ]
        return actions

    def get_temporal_agg_action(self, action_chunk):
        """
        Implements temporal ensembline, as in Aloha ACT. Takes in a current prediction chunk and returns a single ensembled action
        """
        assert action_chunk.shape == (
            self.policy_cfg.action_config.action_chunk_steps,
            self.num_envs,
            self.policy_cfg.action_config.action_dim,
        )

        # Put envs dimension first
        self.all_time_actions[
            :, self.step, self.step : self.step + self.policy_cfg.action_config.action_chunk_steps
        ] = action_chunk.transpose(0, 1)

        actions_for_curr_step = self.all_time_actions[:, :, self.step]

        actions_populated = torch.all(torch.all(actions_for_curr_step != 0, dim=2), dim=0)
        actions_for_curr_step = actions_for_curr_step[:, actions_populated]

        time_indices = torch.arange(
            actions_for_curr_step.shape[1], device=actions_for_curr_step.device, dtype=torch.float
        )
        exp_weights = torch.exp(self.k * time_indices)
        exp_weights = exp_weights / exp_weights.sum()

        weighted_actions = actions_for_curr_step * exp_weights.unsqueeze(-1).unsqueeze(0)

        raw_action = weighted_actions.sum(dim=1)

        return raw_action

    def get_action(self, obs):
        """Returns a single action to be directly executed. For action chunking policies it either uses an previsouly
        predicted action chunk, or if it has exausted all of those actions, it queries the model for a new chunk and returns the first one
        """
        if len(self.action_cache) > 0:
            curr_action = self.action_cache.pop(0)
        else:
            processed_obs = self.process_obs(obs)
            action_chunk = self.predict_action(processed_obs)  # shape: (action_chunk_steps, num_envs, action_dim)
            if self.policy_cfg.action_config.temporal_agg:
                curr_action = self.get_temporal_agg_action(action_chunk)
                curr_action = self.process_action([curr_action], obs)[0]
            else:
                qpos_action = self.process_action(action_chunk, obs)
                assert len(qpos_action) == self.policy_cfg.action_config.action_chunk_steps, (
                    f"Expected {self.policy_cfg.action_config.action_chunk_steps} actions, got {len(qpos_action)}"
                )
                self.action_cache = qpos_action
                curr_action = self.action_cache.pop(0)

        self.step += 1
        assert curr_action.shape == (self.num_envs, len(self.scenario.robots[0].joint_limits.keys())), (
            f"Expected num_envs X n_dof : {self.num_envs} X {len(self.scenario.robots[0].joint_limits.keys())}, got {curr_action.shape} instead"
        )

        actions = self.action_to_dict(curr_action)
        return actions

    def predict_action(self, obs):
        raise NotImplementedError

    def _solve_ik(self, action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q):
        """Solves IK for the given action end-effector action, in either delta or absolute control"""
        assert action.shape == (self.num_envs, self.policy_cfg.action_config.action_dim), (
            f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {action.shape} instead"
        )
        if self.policy_cfg.action_config.ee_cfg.rotation_rep == "quaternion":
            ee_quat_action = action[:, 3:7]
            quat_norm = torch.norm(ee_quat_action, dim=1, keepdim=True)
            ee_quat_action = ee_quat_action / (quat_norm + 1e-5)
        else:
            ee_quat_action = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(action[:, 3:6], "XYZ"))

        if self.policy_cfg.action_config.delta:
            ee_pos_target = curr_ee_pos_local + action[:, :3]
            ee_quat_target = transforms.quaternion_multiply(curr_ee_quat_local, ee_quat_action)
        else:
            ee_pos_target = action[:, :3]
            ee_quat_target = ee_quat_action

        # Solve IK
        seed_config = curr_robot_q[:, : self.curobo_n_dof].unsqueeze(1).tile([1, self.robot_ik._num_seeds, 1])
        result = self.robot_ik.solve_batch(
            Pose(ee_pos_target.cuda(0), ee_quat_target.cuda(0)), seed_config=seed_config.cuda(0)
        )

        if self.policy_cfg.action_config.ee_cfg.gripper_rep == "strength":
            gripper_pos = 1 - action[:, -1]
            gripper_widths = torch.zeros(self.num_envs, self.ee_n_dof, device=self.device)
            for i in range(self.num_envs):
                if gripper_pos[i] < 0.5:
                    gripper_widths[i] = torch.tensor(self.scenario.robots[0].gripper_close_q, device=self.device)
                else:
                    gripper_widths[i] = torch.tensor(self.scenario.robots[0].gripper_open_q, device=self.device)
        else:
            gripper_widths = action[:, -self.ee_n_dof :]

        q = curr_robot_q.clone()
        ik_succ = result.success.squeeze(1).to(self.device)
        if (~ik_succ).any():
            log.warning(f"IK failed: {ik_succ}")
            log.info("Trying to POS delta: ", action[:, :3])

        q[ik_succ, : self.curobo_n_dof] = result.solution.to(self.device)[ik_succ, 0].clone()
        q[:, -self.ee_n_dof :] = gripper_widths
        return q

    def process_action(self, action_chunk, obs):
        """
        Processes a chunk of actions into joint positions.
        """
        action_chunk = [a.to(self.device) for a in action_chunk]
        for a in action_chunk:
            assert a.shape == (self.num_envs, self.policy_cfg.action_config.action_dim), (
                f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {a.shape} instead"
            )
        if self.policy_cfg.action_config.action_type == "joint_pos":
            qpos_action_chunk = action_chunk
        elif self.policy_cfg.action_config.action_type == "ee":
            qpos_action_chunk = []
            robot_ee_state = obs["robot_ee_state"].to(self.device)
            robot_root_state = obs["robot_root_state"].to(self.device)
            robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = transforms.quaternion_apply(
                transforms.quaternion_invert(robot_quat), curr_ee_pos - robot_pos
            )
            curr_ee_quat_local = transforms.quaternion_multiply(transforms.quaternion_invert(robot_quat), curr_ee_quat)
            curr_robot_q = obs["joint_qpos"].to(self.device)
            for action in action_chunk:
                target_qpos = self._solve_ik(action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q)
                qpos_action_chunk.append(target_qpos)

        if self.policy_cfg.action_config.interpolate_chunk:
            return self._interpolate_chunk(obs["joint_qpos"].to(self.device), qpos_action_chunk)
        else:
            return qpos_action_chunk

    def _interpolate_chunk(self, curr_qpos, qpos_action_chunk):
        """Smoothly interpolates between the current state and final predicted action of the chunk"""
        last_action = qpos_action_chunk[-1]
        assert curr_qpos.shape == last_action.shape, (
            f"Expected {curr_qpos.shape} and {last_action.shape} to be the same, got {curr_qpos.shape} and {last_action.shape} instead"
        )

        return [
            curr_qpos + (last_action - curr_qpos) * (i + 1) / self.policy_cfg.action_config.action_chunk_steps
            for i in range(self.policy_cfg.action_config.action_chunk_steps)
        ]

    def reset(self):
        self.action_cache = []
        self.step = 0
