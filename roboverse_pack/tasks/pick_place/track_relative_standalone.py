"""独立的轨迹跟踪任务，假设物体已经被抓取。

手部保持稳定的抓取状态，同时跟随预定义的路径点轨迹。
手部始终使用保存的抓取配置（闭合的手指）。
"""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg, SimParamCfg
from metasim.task.registry import register_task
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.math import matrix_from_quat


@register_task("track_grasp_hand_relative_standalone", "trackgrasphandrelativestandalone")
class TrackGraspHandRelativeStandalone(RLTaskEnv):
    """独立的轨迹跟踪任务，假设物体已经被抓取。

    手部保持稳定的抓取状态，同时跟随预定义的路径点轨迹。
    手部始终使用保存的抓取配置（闭合的手指）。
    """

    HAND_MARKER_NAME = "hand_debug_marker"

    # 保存的手指位置（来自 vega 机器人 L 手手指）- saved_poses_20251126_101518.py
    SAVED_FINGER_POSITIONS = {
        "L_ff_j1": -0.551686,
        "L_ff_j2": -0.621416,
        "L_lf_j1": 0.023751,
        "L_lf_j2": 0.027291,
        "L_mf_j1": -0.551840,
        "L_mf_j2": -0.625254,
        "L_rf_j1": 0.014122,
        "L_rf_j2": 0.015942,
        "L_th_j0": 0.693886,
        "L_th_j1": 0.149071,
        "L_th_j2": 0.201562,
    }

    # 轨迹跟踪配置
    NUM_WAYPOINTS = 5
    REACH_THRESHOLD = 0.10
    ACTION_SCALE = 0.01
    W_TRACKING_APPROACH = 4.0
    W_TRACKING_PROGRESS = 150.0

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="object",
                scale=(0.3, 0.3, 0.3),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="/home/balen/murphy/isaaclab_rv/2/RoboVerse/roboverse_pack/whale_doll/whale_doll.usd",
                urdf_path="roboverse_pack/whale_doll/whale_doll.urdf",
                default_position=(0.104252, -0.076198, 0.846706),
                default_orientation=(0.454115, 0.132146, 0.231502, -0.850132),
            ),
            PrimitiveCubeCfg(
                name="wall",
                size=(0.8, 0.1, 0.3),
                mass=1000.0,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.7, 0.7, 0.7),
                fix_base_link=True,
                default_position=(0.532921, -0.217400, 0.946513),
                default_orientation=(1, 0.0, 0.0, 0.0),
            ),
            RigidObjCfg(
                name="table",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                enabled_gravity=False,
                usd_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/usd/table.usd",
                urdf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/result/table.urdf",
                mjcf_path="roboverse_data/assets/EmbodiedGenData/demo_assets/table/mjcf/table.xml",
                fix_base_link=True,
                default_position=(0.560000, -0.250000, 0.399868),
                default_orientation=(1.000000, -0.000000, -0.000000, 0.000000),
            ),
            # 可视化：轨迹路径点（5个球体显示轨迹路径）
            RigidObjCfg(
                name="traj_marker_0",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_1",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_2",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_3",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name="traj_marker_4",
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
            RigidObjCfg(
                name=HAND_MARKER_NAME,
                urdf_path="roboverse_pack/tasks/pick_place/marker/marker.urdf",
                mjcf_path="roboverse_pack/tasks/pick_place/marker/marker.xml",
                usd_path="roboverse_pack/tasks/pick_place/marker/marker.usd",
                scale=0.2,
                physics=PhysicStateType.XFORM,
                enabled_gravity=False,
                collision_enabled=False,
                fix_base_link=True,
            ),
        ],
        robots=["vega"],
        sim_params=SimParamCfg(
            dt=0.005,
        ),
        decimation=4,
    )
    max_episode_steps = 150

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None
        self._action_scale = self.ACTION_SCALE
        self.num_envs = scenario.num_envs

        # 初始化轨迹跟踪相关变量
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._traj_device = torch.device(device)

        self.num_waypoints = self.NUM_WAYPOINTS
        self.reach_threshold = self.REACH_THRESHOLD
        self.w_tracking_approach = self.W_TRACKING_APPROACH
        self.w_tracking_progress = self.W_TRACKING_PROGRESS

        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self._traj_device)
        self.waypoints_reached = torch.zeros(
            self.num_envs, self.num_waypoints, dtype=torch.bool, device=self._traj_device
        )
        self.prev_distance_to_waypoint = torch.zeros(self.num_envs, device=self._traj_device)

        # 初始化手指位置占位符
        self.saved_finger_targets = None

        # 调用父类初始化
        super().__init__(scenario, device)

        # 初始化手指关节索引
        joint_names = self.handler.get_joint_names(self.robot_name, sort=True)
        self.left_hand_finger_joint_names = [
            "L_th_j0",
            "L_th_j1",
            "L_th_j2",
            "L_ff_j1",
            "L_ff_j2",
            "L_mf_j1",
            "L_mf_j2",
            "L_rf_j1",
            "L_rf_j2",
            "L_lf_j1",
            "L_lf_j2",
        ]
        self.left_hand_finger_joint_indices = [
            joint_names.index(name) for name in self.left_hand_finger_joint_names if name in joint_names
        ]

        # 初始化路径点位置
        initial_states_list = self._get_initial_states()
        if initial_states_list is None or len(initial_states_list) == 0:
            raise ValueError("No initial states found")

        first_env_state = initial_states_list[0]
        waypoint_positions = []

        for i in range(self.num_waypoints):
            marker_name = f"traj_marker_{i}"
            if marker_name in first_env_state["objects"]:
                pos = first_env_state["objects"][marker_name]["pos"]
                waypoint_positions.append(pos)
            else:
                raise ValueError(f"Marker {marker_name} not found in initial states")

        self.waypoint_positions = torch.stack(waypoint_positions).to(device)

        # 初始化奖励函数
        self.reward_functions = [
            self._reward_trajectory_tracking,
        ]
        self.reward_weights = [
            1.0,
        ]

    def reset(self, env_ids=None):
        """重置环境并初始化手指位置。"""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            )

        # 重置轨迹跟踪状态
        self.current_waypoint_idx[env_ids_tensor] = 0
        self.waypoints_reached[env_ids_tensor] = False
        self.prev_distance_to_waypoint[env_ids_tensor] = 0.0

        # 重置动作
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]

        # 调用父类重置
        obs, info = super().reset(env_ids=env_ids)

        # 初始化保存的手指位置
        if self.saved_finger_targets is None:
            num_fingers = len(self.left_hand_finger_joint_indices)
            self.saved_finger_targets = torch.zeros((self.num_envs, num_fingers), device=self.device)
            for i, joint_name in enumerate(self.left_hand_finger_joint_names):
                if i < num_fingers and joint_name in self.SAVED_FINGER_POSITIONS:
                    self.saved_finger_targets[:, i] = self.SAVED_FINGER_POSITIONS[joint_name]

        # 初始化到第一个路径点的距离
        states = self.handler.get_states()
        object_pos = states.objects["object"].root_state[:, 0:3]
        target_pos = self.waypoint_positions[0].unsqueeze(0).expand(len(env_ids), -1)
        self.prev_distance_to_waypoint[env_ids] = torch.norm(object_pos[env_ids] - target_pos, dim=-1)

        return obs, info

    def step(self, actions):
        """步进，使用 delta 控制和手部控制（始终使用保存的手指位置）。"""
        current_states = self.handler.get_states(mode="tensor")

        # 应用 delta 控制
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.clamp(new_actions, self._action_low, self._action_high)

        # 始终使用保存的手指位置（任务假设物体已经被抓取）
        if self.saved_finger_targets is not None:
            for i, joint_idx in enumerate(self.left_hand_finger_joint_indices):
                if i < self.saved_finger_targets.shape[1]:
                    real_actions[:, joint_idx] = self.saved_finger_targets[:, i]

        # 调用父类 step（但需要手动处理，因为我们需要自定义终止条件）
        self._episode_steps += 1

        if not isinstance(real_actions, torch.Tensor):
            real_actions = torch.as_tensor(real_actions, dtype=torch.float32, device=self.device)
        if real_actions.ndim == 1:
            real_actions = real_actions.unsqueeze(0)

        real_actions = torch.maximum(torch.minimum(real_actions, self._action_high), self._action_low)
        self.handler.set_dof_targets(real_actions)
        self.handler.simulate()
        states = self.handler.get_states()
        obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states)
        reward = self._reward(states)
        terminated = self._terminated(states).bool().to(self.device)
        time_out = self._time_out(states).bool().to(self.device)

        # 检查是否所有路径点都已到达 - 如果到达则立即终止
        all_waypoints_reached = self.waypoints_reached.all(dim=1)
        if all_waypoints_reached.any():
            if all_waypoints_reached[0]:
                log.info("[Env 0] All waypoints reached! Terminating episode.")
            terminated = terminated | all_waypoints_reached

        episode_done = terminated | time_out
        info = {
            "privileged_observation": priv_obs,
            "episode_steps": self._episode_steps.clone(),
            "observations": {"raw": {"obs": self._raw_observation_cache.clone()}},
            "stage": torch.full((self.num_envs,), 1, dtype=torch.long, device=self.device),
            "all_waypoints_reached": all_waypoints_reached,
        }

        done_indices = episode_done.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel():
            self.reset(env_ids=done_indices.tolist())
            states_after = self.handler.get_states()
            obs_after = self._observation(states_after).to(self.device)
            obs[done_indices] = obs_after[done_indices]
            self._raw_observation_cache[done_indices] = obs_after[done_indices]
        else:
            keep_mask = (~terminated).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs)

        self._last_action = real_actions

        return obs, reward, terminated, time_out, info

    def _reward_trajectory_tracking(self, env_states) -> torch.Tensor:
        """轨迹跟踪奖励。"""
        # 使用物体位置进行距离计算（物体已经被抓取）
        object_pos = env_states.objects["object"].root_state[:, 0:3]  # (B, 3)
        tracking_reward = torch.zeros(self.num_envs, device=self.device)

        target_pos = self.waypoint_positions[self.current_waypoint_idx]
        distance = torch.norm(object_pos - target_pos, dim=-1)

        approach_reward = (1 - torch.tanh(1.0 * distance)) * self.w_tracking_approach

        reached = distance < self.reach_threshold
        newly_reached = reached & (
            ~self.waypoints_reached[torch.arange(self.num_envs, device=self.device), self.current_waypoint_idx]
        )
        progress_reward = newly_reached.float() * self.w_tracking_progress

        if newly_reached.any():
            if newly_reached[0]:
                wp_idx = self.current_waypoint_idx[0].item()
                log.info(
                    f"[Env 0] Reached waypoint #{wp_idx}! Distance: {distance[0].item():.4f}m < {self.reach_threshold}m"
                )

            self.waypoints_reached[newly_reached, self.current_waypoint_idx[newly_reached]] = True

            can_advance = newly_reached & (self.current_waypoint_idx < self.num_waypoints - 1)

            if can_advance.any() and can_advance[0]:
                old_idx = self.current_waypoint_idx[0].item()
                new_idx = old_idx + 1
                log.info(f"   -> Advancing to waypoint #{new_idx}")

            self.current_waypoint_idx[can_advance] += 1

            if can_advance.any():
                new_target_pos = self.waypoint_positions[self.current_waypoint_idx[can_advance]]
                self.prev_distance_to_waypoint[can_advance] = torch.norm(
                    object_pos[can_advance] - new_target_pos, dim=-1
                )

        maintain_reward = torch.zeros(self.num_envs, device=self.device)
        all_reached = self.waypoints_reached.all(dim=1)

        if all_reached.any():
            last_target_pos = self.waypoint_positions[-1].unsqueeze(0).expand(self.num_envs, -1)
            distance_to_last = torch.norm(object_pos - last_target_pos, dim=-1)

            maintain_reward[all_reached] = torch.where(
                distance_to_last[all_reached] < self.reach_threshold,
                torch.full((all_reached.sum(),), 5, device=self.device),
                (1 - torch.tanh(1.0 * distance_to_last[all_reached])) * self.w_tracking_approach,
            )

        tracking_reward = torch.where(all_reached, maintain_reward, approach_reward + progress_reward)

        return tracking_reward

    def _get_hand_position(self, states):
        """获取手掌中心位置。"""
        rs = states.robots[self.robot_name]
        body_state = rs.body_state
        if body_state is None:
            raise ValueError("Robot body_state is required to compute hand pose.")
        if not torch.is_tensor(body_state):
            body_state = torch.as_tensor(body_state, device=self.device, dtype=torch.float32)
        else:
            body_state = body_state.to(self.device)

        name_to_index = {name: idx for idx, name in enumerate(rs.body_names)}
        if "L_arm_l7" not in name_to_index:
            raise ValueError("Required link 'L_arm_l7' missing in body_names.")
        link_index = name_to_index["L_arm_l7"]

        link_pos = body_state[:, link_index, 0:3]
        link_quat = body_state[:, link_index, 3:7]

        hand_offset_local = torch.tensor(
            [0.25864, -0.035, -0.03513],
            device=link_pos.device,
            dtype=link_pos.dtype,
        )
        hand_offset = (
            hand_offset_local.view(1, 3, 1)
            .repeat(link_pos.shape[0], 1, 1)
        )
        link_rot = matrix_from_quat(link_quat)
        hand_pos = link_pos + torch.bmm(link_rot, hand_offset).squeeze(-1)

        return hand_pos

    def _observation(self, env_states) -> torch.Tensor:
        """获取观察值。"""
        object_pos = env_states.objects["object"].root_state[:, 0:3]  # [num_envs, 3]
        object_quat = env_states.objects["object"].root_state[:, 3:7]  # [num_envs, 4]

        hand_pos = self._get_hand_position(env_states)  # (B, 3)

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # 将四元数转换为旋转矩阵
        box_mat = matrix_from_quat(object_quat)  # [num_envs, 3, 3]
        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]

        object_to_hand = object_pos - hand_pos  # [num_envs, 3]

        target_pos = self.waypoint_positions[self.current_waypoint_idx]
        target_to_hand = target_pos - hand_pos
        distance_to_target = torch.norm(target_to_hand, dim=-1, keepdim=True)

        waypoint_onehot = torch.nn.functional.one_hot(self.current_waypoint_idx, num_classes=self.num_waypoints).float()

        num_reached = self.waypoints_reached.sum(dim=1, keepdim=True).float() / self.num_waypoints

        obs_list = [
            robot_joint_pos,
            robot_joint_vel,
            hand_pos,  # 手部基础位置
            box_mat_flat[:, 3:],
            object_to_hand,
            target_pos,
            target_to_hand,
            distance_to_target,
            waypoint_onehot,
            num_reached,
        ]

        obs = torch.cat(obs_list, dim=-1)  # [num_envs, obs_dim]

        return obs

    def _privileged_observation(self, env_states) -> torch.Tensor:
        """获取特权观察值（与普通观察值相同）。"""
        return self._observation(env_states)

    def _terminated(self, env_states) -> torch.Tensor:
        """终止标志（默认：无）。"""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _time_out(self, env_states) -> torch.Tensor:
        """超时标志。"""
        return self._episode_steps >= self.max_episode_steps

    def _get_initial_states(self) -> list[dict] | None:
        """获取所有环境的初始状态。"""
        init = [
            {
                "objects": {
                    "object": {
                        "pos": torch.tensor([0.104252, -0.076198, 0.846706]),
                        "rot": torch.tensor([0.454115, 0.132146, 0.231502, -0.850132]),
                    },
                    "wall": {
                        "pos": torch.tensor([0.532921, -0.217400, 0.946513]),
                        "rot": torch.tensor([0.999490, -0.000045, 0.001448, -0.031900]),
                    },
                    "table": {
                        "pos": torch.tensor([0.560000, -0.250000, 0.399868]),
                        "rot": torch.tensor([1.000000, -0.000000, -0.000000, 0.000000]),
                    },
                    # 轨迹路径点（世界坐标）
                    "traj_marker_0": {
                        "pos": torch.tensor([0.300000, -0.460000, 1.020000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_1": {
                        "pos": torch.tensor([0.300000, -0.320000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_2": {
                        "pos": torch.tensor([0.300000, -0.190000, 1.220000]),
                        "rot": torch.tensor([0.998750, 0.000000, 0.049979, 0.000000]),
                    },
                    "traj_marker_3": {
                        "pos": torch.tensor([0.300000, -0.070000, 1.220000]),
                        "rot": torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]),
                    },
                    "traj_marker_4": {
                        "pos": torch.tensor([0.300000, 0.000000, 1.080000]),
                        "rot": torch.tensor([0.984726, 0.000000, 0.174108, 0.000000]),
                    },
                    self.HAND_MARKER_NAME: {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "vega": {
                        "pos": torch.tensor([-0.353636, -0.230209, 0.000511]),
                        "rot": torch.tensor([1.000201, 0.000000, -0.000000, -0.000000]),
                        "dof_pos": {
                            # 来自 saved_poses_20251126_101518.py
                            "B_wheel_j1": 3.098256,
                            "B_wheel_j2": 11.410545,
                            "L_arm_j1": 0.132269,
                            "L_arm_j2": -0.005679,
                            "L_arm_j3": -0.039935,
                            "L_arm_j4": -0.007740,
                            "L_arm_j5": 0.792566,
                            "L_arm_j6": -0.068200,
                            "L_arm_j7": 0.003688,
                            "L_ff_j1": -0.551686,
                            "L_ff_j2": -0.621416,
                            "L_lf_j1": 0.023751,
                            "L_lf_j2": 0.027291,
                            "L_mf_j1": -0.551840,
                            "L_mf_j2": -0.625254,
                            "L_rf_j1": 0.014122,
                            "L_rf_j2": 0.015942,
                            "L_th_j0": 0.693886,
                            "L_th_j1": 0.149071,
                            "L_th_j2": 0.201562,
                            "L_wheel_j1": -0.050129,
                            "L_wheel_j2": -2.361024,
                            "R_arm_j1": -0.012039,
                            "R_arm_j2": 0.000147,
                            "R_arm_j3": 0.002277,
                            "R_arm_j4": 0.015811,
                            "R_arm_j5": -0.001019,
                            "R_arm_j6": -0.003106,
                            "R_arm_j7": 0.000569,
                            "R_ff_j1": 0.006052,
                            "R_ff_j2": 0.006850,
                            "R_lf_j1": 0.008481,
                            "R_lf_j2": 0.009723,
                            "R_mf_j1": 0.005918,
                            "R_mf_j2": 0.006720,
                            "R_rf_j1": 0.007117,
                            "R_rf_j2": 0.008026,
                            "R_th_j0": 0.264549,
                            "R_th_j1": 0.078753,
                            "R_th_j2": 0.106434,
                            "R_wheel_j1": 0.081973,
                            "R_wheel_j2": 5.777865,
                            "torso_j1": 0.153865,
                            "torso_j2": 0.016780,
                            "torso_j3": -0.021553,
                        },
                    },
                },
            }
            for _ in range(self.num_envs)
        ]

        return init









