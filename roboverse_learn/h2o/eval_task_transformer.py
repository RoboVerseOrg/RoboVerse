import argparse
import os
import re
import subprocess

parser = argparse.ArgumentParser(description="Run eval_hydra_BC_policy.py on multiple ckpt files.")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the directory containing ckpt files.")
parser.add_argument("--motion_file", type=str, required=True, help="Motion parameter value.")
parser.add_argument("--num_envs", type=int, required=True, help="Num Envs.")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI).")
parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run with GUI (not headless).")
parser.set_defaults(headless=True)

args = parser.parse_args()

# 提取参数
CKPT_PATH = args.ckpt_path
MOTION_FILE = args.motion_file
NUM_ENVS = args.num_envs
# 定义路径和其他固定参数
# CKPT_PATH = "/home/yunshen/code/test_ckpt/amass_200k_mix@amass_100k_easy_clean/"
SCRIPT_PATH = "legged_gym/scripts/eval_hydra_BC_policy.py"
CONFIG_NAME = "config_teleop_humanoid_data_gene_student_obs_for_play_8_4_transformer_15_step_x0_delay_data_8_8_256_0"
TASK = "h1:teleop"
ENV_NUM_OBSERVATIONS = 913
ENV_NUM_PRIVILEGED_OBS = 990
MOTION_FUTURE_TRACKS = "True"
MOTION_TELEOP_OBS_VERSION = "v-teleop-extend-max-full"
MOTION = "motion_full"
MOTION_EXTEND_HEAD = "True"
ASSET_ZERO_OUT_FAR = "False"
ASSET_TERMINATION_SCALES_MAX_REF_MOTION_DISTANCE = 1.0
SIM_DEVICE = "cuda:0"
LOAD_RUN = "24_10_10_18-52-15_OmniH2O_TEACHER"
CHECKPOINT = 555000

HEADLESS = args.headless
REWARDS = "rewards_teleop_omnih2o_teacher"
# MOTION_FILE = "resources/motions/h1/kit_6.pkl"
PLAY_IN_ORDER = "False"
LOG_ROOT = "default"


def extract_step_number(file_name):
    match = re.search(r"step_(\d+)", file_name)
    return int(match.group(1)) if match else float("inf")  # 无匹配时返回无穷大，确保排在最后


# 获取所有文件的完整路径并排序
ckpt_files = [os.path.join(CKPT_PATH, f) for f in os.listdir(CKPT_PATH) if f.endswith(".ckpt")]
sorted_ckpt_files = sorted(ckpt_files, key=extract_step_number)

# 过滤步数，保留 10, 20, 30... 这样的文件
filtered_ckpt_files = [f for f in sorted_ckpt_files if extract_step_number(f) % 10 == 0]

# 遍历每个 ckpt 文件并执行命令
for ckpt_file in sorted_ckpt_files:
    command = [
        "python",
        SCRIPT_PATH,
        f"--config-name={CONFIG_NAME}",
        f"task={TASK}",
        f"env.num_observations={ENV_NUM_OBSERVATIONS}",
        f"env.num_privileged_obs={ENV_NUM_PRIVILEGED_OBS}",
        f"motion.teleop_obs_version={MOTION_TELEOP_OBS_VERSION}",
        f"motion={MOTION}",
        f"motion.extend_head={MOTION_EXTEND_HEAD}",
        f"asset.zero_out_far={ASSET_ZERO_OUT_FAR}",
        f"asset.termination_scales.max_ref_motion_distance={ASSET_TERMINATION_SCALES_MAX_REF_MOTION_DISTANCE}",
        f"sim_device={SIM_DEVICE}",
        f"load_run={LOAD_RUN}",
        f"checkpoint={CHECKPOINT}",
        f"num_envs={NUM_ENVS}",
        f"headless={HEADLESS}",
        f"rewards={REWARDS}",
        f"motion.motion_file={MOTION_FILE}",
        f"play_in_order={PLAY_IN_ORDER}",
        f"log_root={LOG_ROOT}",
        f"BC_ckpt_path={ckpt_file}",
        f"motion.future_tracks={MOTION_FUTURE_TRACKS}",
    ]

    # 打印命令以供调试
    print("Running command:", " ".join(command))

    # 执行命令
    subprocess.run(command, check=True)
