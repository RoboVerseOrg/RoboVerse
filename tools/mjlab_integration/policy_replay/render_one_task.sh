#!/usr/bin/env bash
# Render side-by-side policy rollout for one mjlab task.
#
# Usage:
#   bash render_one_task.sh <task-id> <ckpt> [motion]
#
# Produces:
#   <RUNS_DIR>/<task-metasim-name>/policy_side_by_side.mp4
#   (and intermediate: mjlab_native.mp4, metasim_replay.mp4, trajectory.npz, metasim_replay.npz)

set -euo pipefail

if command -v conda &>/dev/null && [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate roboverse
fi

REPO="${ROBOVERSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)}"
MJLAB="${MJLAB_DIR:-$HOME/projects/mjlab}"
RUNS="${ROBOVERSE_REPORTS_DIR:-$REPO/outputs/reports}/mjlab_integration/runs"

TASK="${1:?task id required, e.g. Mjlab-Tracking-Flat-Unitree-G1}"
CKPT="${2:?ckpt path required}"
MOTION="${3:-}"
STEPS="${STEPS:-400}"
HEIGHT="${HEIGHT:-360}"
WIDTH="${WIDTH:-640}"
NCONMAX="${NCONMAX:-}"  # optional override; rough-terrain needs ≥256

# Derive metasim name (Mjlab-Tracking-Flat-Unitree-G1 → mjlab.tracking_flat_g1)
METASIM_NAME=$(python3 -c "
import sys, os
sys.path.insert(0, '$REPO')
from tools.mjlab_integration import inventory as inv
for t in inv.ALL_TASKS:
    if t.task_id == '$TASK':
        print(t.metasim_name); break
")
if [ -z "$METASIM_NAME" ]; then
    echo "ERROR: task '$TASK' not in inventory" >&2; exit 1
fi
OUT="$RUNS/$METASIM_NAME"
echo "[render_one_task] task=$TASK  metasim=$METASIM_NAME"
echo "[render_one_task] out=$OUT"
echo "[render_one_task] ckpt=$CKPT"
[ -n "$MOTION" ] && echo "[render_one_task] motion=$MOTION"

# 1) Run mjlab native + record trajectory + left mp4
MOTION_ARG=()
[ -n "$MOTION" ] && MOTION_ARG=(--motion "$MOTION")
NCONMAX_ARG=()
[ -n "$NCONMAX" ] && NCONMAX_ARG=(--nconmax "$NCONMAX")
cd "$MJLAB"
PYTHONPATH="$REPO" MUJOCO_GL=egl python -u -m tools.mjlab_integration.policy_replay.run_mjlab_policy \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    "${MOTION_ARG[@]}" \
    "${NCONMAX_ARG[@]}" \
    --steps "$STEPS" \
    --height "$HEIGHT" --width "$WIDTH" \
    --out "$OUT"

# 2) Replay in MetaSim/MuJoCo + render right mp4
cd "$REPO"
PYTHONPATH="$REPO" MUJOCO_GL=egl python -u -m tools.mjlab_integration.policy_replay.replay_in_metasim \
    --task "$TASK" \
    --trajectory "$OUT/trajectory.npz" \
    --width "$WIDTH" --height "$HEIGHT" \
    --out "$OUT"

# 3) Stitch side-by-side
PYTHONPATH="$REPO" python3 -m tools.mjlab_integration.policy_replay.stitch_side_by_side \
    --left "$OUT/mjlab_native.mp4" \
    --right "$OUT/metasim_replay.mp4" \
    --out "$OUT/policy_side_by_side.mp4" \
    --height "$HEIGHT"

echo "[render_one_task] DONE: $OUT/policy_side_by_side.mp4"
ls -la "$OUT/" | grep mp4
