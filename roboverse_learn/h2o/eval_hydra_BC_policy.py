import sys
import os
import csv
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import task_registry, Logger
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from rsl_rl.runners.eval_runner_BC_modified import EvalRunnerBCModified
import json
NOROSPY = False
try:
    import rospy
except:
    NOROSPY = True
# from std_msgs.msg import String, Header, Float64MultiArray



override = False

EXPORT_ONNX = True

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config_base",
)
def play(cfg_hydra: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg_hydra))
    cfg_humanoid_workspace = cfg_hydra.humanoid_workspace
    cfg_humanoid_dataset = cfg_hydra.humanoid_workspace.task.dataset
    cfg_humanoid_dataloer = cfg_hydra.humanoid_workspace.dataloader
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    env_cfg, train_cfg = cfg_hydra, cfg_hydra.train

    # if not env_cfg.train_velocity_estimation:
    env_cfg.env.num_envs = 1
    env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.mesh_type = 'plane'
    # if env_cfg.terrain.mesh_type == 'trimesh':
    #     env_cfg.terrain.terrain_types = ['flat', 'rough', 'low_obst']  # do not duplicate!
    #     env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0]
    env_cfg.add_eval_noise= False
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level= 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20
    env_cfg.domain_rand.randomize_rfi_lim = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.link_mass_range = [0.7, 1.3]
    # env_cfg.domain_rand.randomize_link_body_names = [
    #     'pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link',
    #     'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link',
    #     'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    #     'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
    # ]
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [1, 2]
    clip_action = True


    # env_cfg.asset.termination_scales.max_ref_motion_distance = 1



    env_cfg.env.test = True

    if env_cfg.motion.realtime_vr_keypoints:
        env_cfg.asset.terminate_by_1time_motion = False
        env_cfg.asset.terminate_by_ref_motion_distance = False
        rospy.init_node("avppose_subscriber")
        from avp_pose_subscriber import AVPPoseInfo
        avpposeinfo = AVPPoseInfo()
        rospy.Subscriber("avp_pose", Float64MultiArray, avpposeinfo.avp_callback, queue_size=1)
    if cfg_hydra.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard
        from legged_gym.utils import key_response_fn

    # prepare environment

    env, _ = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=env_cfg)
    # env.compute_observations()
    # import ipdb;ipdb.set_trace()
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    # #import ipdb;ipdb.set_trace()
    # obs = env.compute_observations()
    # import ipdb;ipdb.set_trace()
    # obs_student = env.obs_student_buf.clone() #must get_observation() first, then we can get the latest obs_student_buf
    To = cfg_humanoid_workspace.n_obs_steps
    # obs_window = []
    # #import ipdb;ipdb.set_trace()
    # for i in range(To):
    #     obs_window.append(obs)
    # obs_window_init = obs_window.copy()
    # if env_cfg.motion.realtime_vr_keypoints:
    #     init_root_pos = env._rigid_body_pos[..., 0, :].clone()
    #     init_avp_pos = avpposeinfo.avp_pose.copy()
    #     init_root_offset = init_root_pos[0, :2] - init_avp_pos[2, :2]
    # import ipdb; ipdb.set_trace()
    # obs[:, 9:12] = torch.Tensor([0.5, 0, 0])
    # load policy
    train_cfg.runner.resume = True

    humanoid_workspace = task_registry.load_BC_workspace(checkpoint_path=cfg_hydra.BC_ckpt_path, train_cfg=cfg_humanoid_workspace)
    #ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=train_cfg,log_root=cfg_hydra.log_root)
    policy = humanoid_workspace.model.to(env.device)
    eval_runner = EvalRunnerBCModified(env=env,policy=policy,train_cfg=train_cfg,device=env.device,To=To, clip_action = clip_action)

    results = eval_runner.eval()
    ckpt_path = cfg_hydra.BC_ckpt_path  # 例如 "/home/yunshen/code/test_ckpt/amass_200k_mix@amass_100k_easy_clean/epoch_400_step_663387.ckpt"
    file_name = os.path.basename(os.path.dirname(ckpt_path))
    motion_file = os.path.splitext(os.path.basename(cfg_hydra.motion.motion_file))[0]  # 提取 motion 文件名并去掉 ".pkl"
    annotation = os.path.basename(ckpt_path).replace(".ckpt", "")  # 保持原始 annotation 格式



    output_dir = "./eval_results"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"{motion_file}_{os.path.basename(os.path.dirname(ckpt_path))}.csv")
    # 将 results 转换为 JSON 可序列化的格式
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):  # 如果是 numpy 数组，转换为列表
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):  # 如果是 numpy 浮点数，转换为原生浮点数
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):  # 如果是 numpy 整数，转换为原生整数
            return int(obj)
        elif isinstance(obj, dict):  # 如果是字典，递归处理
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # 如果是列表，递归处理
            return [convert_to_serializable(v) for v in obj]
        else:  # 其他类型直接返回（可能是原生类型）
            return obj

    serializable_results = convert_to_serializable(results)

    # 将数据写入 CSV 文件
    header = ["ckpt"] + list(serializable_results.keys())  # 表头，包括 "ckpt" 和结果字段

    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # 写入表头
        row = [annotation] + list(serializable_results.values())  # 每一行数据
        writer.writerow(row)  # 追加数据

    print(f"Results appended to {results_file}")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    # args = get_args()
    play()
