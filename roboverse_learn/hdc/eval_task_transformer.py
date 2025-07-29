#!/usr/bin/env python3
# Run eval_hydra_BC_policy.py over a folder of checkpoints.

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import argparse
import os
import re
import subprocess

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Batch-evaluate BC ckpts with eval_hydra_BC_policy.py")
parser.add_argument("--ckpt_path", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

# ---------------------------------------------------------------------
# Fixed overrides
# ---------------------------------------------------------------------
# SCRIPT_PATH = "roboverse_learn/hdc/eval_hydra_BC_policy.py"
# CONFIG_NAME = "config_teleop_humanoid_data_gene_student_obs_for_play_8_4_transformer_15_step_x0_delay_data_8_8_256_0"
# TASK = "h1:teleop"
# ENV_NUM_OBS = 913
# ENV_NUM_PRIV_OBS = 990
# SIM_DEVICE = "cuda:0"
# LOAD_RUN = "24_10_10_18-52-15_OmniH2O_TEACHER"
# CHECKPOINT = 555000
# REWARDS = "rewards_teleop_omnih2o_teacher"

CKPT_PATH = args.ckpt_path
MOTION_FILE = args.motion_file
NUM_ENVS = args.num_envs
# 定义路径和其他固定参数
#CKPT_PATH = "/home/yunshen/code/test_ckpt/amass_200k_mix@amass_100k_easy_clean/"
SCRIPT_PATH = "roboverse_learn/hdc/eval_hydra_BC_policy.py"
# SCRIPT_PATH = "legged_gym/legged_gym/scripts/eval_hydra_BC_policy.py"
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

HEADLESS = True
REWARDS = "rewards_teleop_omnih2o_teacher"
#MOTION_FILE = "resources/motions/h1/kit_6.pkl"
PLAY_IN_ORDER = "False"
LOG_ROOT = "default"

# ---------------------------------------------------------------------
def step_num(fname: str) -> int:
    m = re.search(r"step_(\d+)", fname)
    return int(m.group(1)) if m else 10**12


ckpts = sorted(
    [os.path.join(args.ckpt_path, f) for f in os.listdir(args.ckpt_path) if f.endswith(".ckpt")],
    key=step_num,
)

for ck in ckpts:
    cmd =  [
        "python", SCRIPT_PATH,
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
        f"BC_ckpt_path={ck}",
        f"motion.future_tracks={MOTION_FUTURE_TRACKS}",

    ]

    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)
