#!/usr/bin/env python3
# Run eval_hydra_BC_policy.py over a folder of checkpoints.

import os
import re
import subprocess
import argparse


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Batch-evaluate BC ckpts with eval_hydra_BC_policy.py")
parser.add_argument("--ckpt_path",   required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--num_envs",    type=int, required=True)
parser.add_argument("--headless",    action="store_true")
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

# ---------------------------------------------------------------------
# Fixed overrides
# ---------------------------------------------------------------------
SCRIPT_PATH = "roboverse_learn/h2o/eval_hydra_BC_policy.py"
CONFIG_NAME      = "config_teleop_humanoid_data_gene_student_obs_for_play_8_4_transformer_15_step_x0_delay_data_8_8_256_0"
TASK             = "h1:teleop"
ENV_NUM_OBS      = 913
ENV_NUM_PRIV_OBS = 990
SIM_DEVICE       = "cuda:0"
LOAD_RUN         = "24_10_10_18-52-15_OmniH2O_TEACHER"
CHECKPOINT       = 555000
REWARDS          = "rewards_teleop_omnih2o_teacher"

# extra teleop-specific flags
EXTRA_OVERRIDES = dict(
    motion_teleop_obs_version="v-teleop-extend-max-full",
    motion="motion_full",
    motion_extend_head="True",
    asset_zero_out_far="False",
    asset_termination_scales_max_ref_motion_distance=1.0,
    play_in_order="False",
    motion_future_tracks="True",
    log_root="default",
)

# ---------------------------------------------------------------------
def step_num(fname: str) -> int:
    m = re.search(r"step_(\d+)", fname)
    return int(m.group(1)) if m else 10**12


ckpts = sorted(
    [os.path.join(args.ckpt_path, f) for f in os.listdir(args.ckpt_path)
     if f.endswith(".ckpt")],
    key=step_num,
)

for ck in ckpts:
    cmd = [
        "python", SCRIPT_PATH,
        f"--config-name={CONFIG_NAME}",
        f"task={TASK}",
        f"env.num_observations={ENV_NUM_OBS}",
        f"env.num_privileged_obs={ENV_NUM_PRIV_OBS}",
        f"sim_device={SIM_DEVICE}",
        f"load_run={LOAD_RUN}",
        f"checkpoint={CHECKPOINT}",
        f"num_envs={args.num_envs}",
        f"headless={args.headless}",
        f"rewards={REWARDS}",
        f"motion.motion_file={args.motion_file}",
        f"BC_ckpt_path={ck}",
    ]
    # append extra fixed overrides
    cmd.extend(f"{k.replace('_', '.')}={v}" for k, v in EXTRA_OVERRIDES.items())

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
