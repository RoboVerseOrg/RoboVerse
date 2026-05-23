#!/usr/bin/env bash
# Sequential training pipeline for mjlab tasks lacking checkpoints.
#
# For each task in the list: train via mjlab.scripts.train, then render the
# policy side-by-side comparison (mjlab native vs MetaSim action-replay).
#
# Skips tasks that already have a side-by-side mp4 in the run dir.
#
# Designed to run unattended in the background. Resumable: if a stage fails,
# the next invocation picks up at the next task.

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate roboverse

REPO="${ROBOVERSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)}"
MJLAB="${MJLAB_DIR:-$HOME/projects/mjlab}"
LOG_ROOT="$REPO/training_logs/rsl_rl"
RUNS="${ROBOVERSE_REPORTS_DIR:-$REPO/outputs/reports}/mjlab_integration/runs"
RENDER_SCRIPT="$REPO/tools/mjlab_integration/policy_replay/render_one_task.sh"

# Tunable per-task. Format: "task_id|experiment_name|max_iter|num_envs|steps|extra_args"
MOTION="${MOTION:-/tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz}"
TASKS=(
  "Mjlab-Cartpole-Swingup|cartpole_swingup|400|4096|400|"
  "Mjlab-Velocity-Flat-Unitree-Go1|go1_velocity_flat|2000|4096|400|"
  "Mjlab-Velocity-Rough-Unitree-Go1|go1_velocity_rough|2000|4096|400|"
  "Mjlab-Velocity-Rough-Unitree-G1|g1_velocity_rough|2000|4096|400|"
  "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation|g1_tracking_no_state_est|2000|4096|400|--env.commands.motion.motion-file ${MOTION}"
  "Mjlab-Lift-Cube-Yam|yam_lift_cube|2000|4096|200|"
  "Mjlab-Multi-Cube-Seg-Yam|yam_multi_cube|2000|4096|200|"
  "Mjlab-Lift-Cube-Yam-Rgb|yam_lift_cube_rgb|2000|4096|200|"
  "Mjlab-Lift-Cube-Yam-Depth|yam_lift_cube_depth|2000|4096|200|"
)

train_one() {
    local task="$1" expname="$2" maxiter="$3" numenvs="$4" extra="$5"
    local exp_dir="$LOG_ROOT/$expname"
    mkdir -p "$exp_dir"
    # Skip if already have a recent ckpt at the requested iter.
    local latest_ckpt=$(ls -1 "$exp_dir"/*/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$latest_ckpt" ]; then
        local n=$(basename "$latest_ckpt" .pt | sed 's/model_//')
        if [ "${n:-0}" -ge "$((maxiter - 1))" ]; then
            echo "[train_all] $task: ckpt already at iter $n (≥ $maxiter-1), skipping train"
            echo "$latest_ckpt"
            return 0
        fi
    fi
    echo "[train_all] training $task → $exp_dir  (max-iter=$maxiter, num-envs=$numenvs)"
    local log="$LOG_ROOT/${expname}_train.log"
    cd "$MJLAB"
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -u -m mjlab.scripts.train "$task" \
        --video False \
        --log-root "$LOG_ROOT" \
        --agent.experiment-name "$expname" \
        --agent.max-iterations "$maxiter" \
        --agent.logger tensorboard \
        --env.scene.num-envs "$numenvs" \
        $extra > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[train_all] $task FAILED (rc=$rc). See $log"
        tail -20 "$log"
        return $rc
    fi
    local ckpt=$(ls -1 "$exp_dir"/*/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$ckpt" ]; then
        echo "[train_all] $task: no ckpt found after train"
        return 1
    fi
    echo "$ckpt"
}

render_one() {
    local task="$1" ckpt="$2" steps="$3"
    local metasim_name=$(python3 -c "
import sys; sys.path.insert(0, '$REPO')
from tools.mjlab_integration import inventory as inv
for t in inv.ALL_TASKS:
    if t.task_id == '$task': print(t.metasim_name); break
")
    local out="$RUNS/$metasim_name"
    if [ -f "$out/policy_side_by_side.mp4" ]; then
        local age_sec=$(($(date +%s) - $(stat -c%Y "$out/policy_side_by_side.mp4")))
        if [ "$age_sec" -lt 86400 ]; then
            echo "[train_all] $task: policy_side_by_side.mp4 already exists ($age_sec s old), skipping render"
            return 0
        fi
    fi
    echo "[train_all] rendering $task → $out"
    # Tracking-* tasks need a motion file at render time too (not just training)
    local render_motion=""
    if [[ "$task" == *Tracking* ]]; then
        render_motion="$MOTION"
    fi
    # Rough-* tasks need bumped nconmax for mujoco-warp contact buffer
    local render_nconmax=""
    if [[ "$task" == *Rough* ]]; then
        render_nconmax="1024"
    fi
    STEPS="$steps" HEIGHT=360 WIDTH=640 MOTION="$render_motion" NCONMAX="$render_nconmax" \
        bash "$RENDER_SCRIPT" "$task" "$ckpt" "$render_motion"
}

for row in "${TASKS[@]}"; do
    IFS='|' read -r task expname maxiter numenvs steps extra <<<"$row"
    echo ""
    echo "============================================================"
    echo "  $task"
    echo "============================================================"
    ckpt=$(train_one "$task" "$expname" "$maxiter" "$numenvs" "$extra" | tail -1)
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "[train_all] $task: no ckpt produced, skipping render"
        continue
    fi
    render_one "$task" "$ckpt" "$steps" || echo "[train_all] $task: render failed, continuing"
done

echo ""
echo "[train_all] pipeline complete"
