#!/usr/bin/env bash
CKPT_DIR=""
MOTION_FILE=""
NUM_ENVS=2048            # Parallel environments
HEADLESS=true            # true = headless, false = GUI
STEP_FILTER_MOD=10       # Keep ckpts where step % MOD == 0 (set 1 to disable)
CSV_DIR="./eval_results" # Folder for CSV output


COMMON_OVERRIDES="
task=h1:teleop
env.num_observations=913
env.num_privileged_obs=990
motion.future_tracks=True
motion.teleop_obs_version=v-teleop-extend-max-full
motion=motion_full
motion.extend_head=True
asset.zero_out_far=False
asset.termination_scales.max_ref_motion_distance=1.0
sim_device=cuda:0
load_run=24_10_10_18-52-15_OmniH2O_TEACHER
checkpoint=555000
rewards=rewards_teleop_omnih2o_teacher
play_in_order=False
log_root=default
"

# ---------- Create output directory ----------
mkdir -p "$CSV_DIR"
MOTION_STUB=$(basename "$MOTION_FILE" .pkl)
CSV_PATH="$CSV_DIR/${MOTION_STUB}_$(basename "$CKPT_DIR").csv"
[[ -f "$CSV_PATH" ]] || echo "Target CSV: $CSV_PATH (will be created)."

# ---------- Iterate over checkpoints ----------
for ckpt in "$CKPT_DIR"/*.ckpt; do
    step=$(echo "$ckpt" | grep -oP 'step_\K[0-9]+')
    [[ -z "$step" ]] && step=inf
    (( step % STEP_FILTER_MOD == 0 )) || continue

    echo "=== Evaluating $(basename "$ckpt") ==="
    python eval_BC_single.py \
        ckpt="$ckpt" \
        motion.motion_file="$MOTION_FILE" \
        num_envs="$NUM_ENVS" \
        headless="$HEADLESS" \
        csv_out="$CSV_PATH" \
        $COMMON_OVERRIDES
done
