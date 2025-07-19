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
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [1, 2]
    clip_action = True



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
    print(OmegaConf.to_yaml(env_cfg))
    env, _ = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=env_cfg)
    To = cfg_humanoid_workspace.n_obs_steps
    train_cfg.runner.resume = True

    humanoid_workspace = task_registry.load_BC_workspace(checkpoint_path=cfg_hydra.BC_ckpt_path, train_cfg=cfg_humanoid_workspace)
    policy = humanoid_workspace.model.to(env.device)
    eval_runner = EvalRunnerBCModified(env=env,policy=policy,train_cfg=train_cfg,device=env.device,To=To, clip_action = clip_action)

    results = eval_runner.eval()
    ckpt_path = cfg_hydra.BC_ckpt_path  # 例如 "/home/yunshen/code/test_ckpt/amass_200k_mix@amass_100k_easy_clean/epoch_400_step_663387.ckpt"
    motion_file = os.path.splitext(os.path.basename(cfg_hydra.motion.motion_file))[0]  # 提取 motion 文件名并去掉 ".pkl"
    annotation = os.path.basename(ckpt_path).replace(".ckpt", "")  # 保持原始 annotation 格式



    output_dir = "./eval_results"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"{motion_file}_{os.path.basename(os.path.dirname(ckpt_path))}.csv")
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)
    header = ["ckpt"] + list(serializable_results.keys())

    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        row = [annotation] + list(serializable_results.values())
        writer.writerow(row)

    print(f"Results appended to {results_file}")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    play()
