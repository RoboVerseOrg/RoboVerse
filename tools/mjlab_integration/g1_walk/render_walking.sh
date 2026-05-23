#!/usr/bin/env bash
# Render G1 walking mp4 from a freshly-trained Mjlab-Velocity-Flat-Unitree-G1 checkpoint.
#
# Usage:
#   bash render_walking.sh                                # uses latest ckpt
#   bash render_walking.sh /path/to/model_2000.pt         # explicit ckpt
#
# Output: reports/mjlab_integration/runs/g1_walking/g1_walking.mp4
#
# Notes:
# - Uses run_pretrained.py with --task Mjlab-Velocity-Flat-Unitree-G1 (no
#   motion file needed; velocity task has no MotionCommandCfg).
# - 1000 steps × 720p. ≈30 s mp4.

set -euo pipefail

# Activate conda env (idempotent; safe to re-source).
if command -v conda &>/dev/null && [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate roboverse
fi

REPO_ROOT="${ROBOVERSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)}"
MJLAB_DIR="${MJLAB_DIR:-$HOME/projects/mjlab}"
LOG_ROOT="${REPO_ROOT}/training_logs/rsl_rl/g1_velocity"
OUT_DIR="${ROBOVERSE_REPORTS_DIR:-$REPO_ROOT/outputs/reports}/mjlab_integration/runs/g1_walking"

# Find latest run dir under g1_velocity/ if no explicit ckpt arg.
if [ "$#" -ge 1 ]; then
    CKPT="$1"
else
    LATEST_RUN=$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1)
    if [ -z "${LATEST_RUN}" ]; then
        echo "ERROR: no run found under ${LOG_ROOT}" >&2
        exit 1
    fi
    # Pick the largest model_NNN.pt by iteration number.
    CKPT=$(ls -1 "${LATEST_RUN}"/model_*.pt 2>/dev/null \
            | awk -F'model_|\\.pt' '{print $2, $0}' \
            | sort -n | tail -n 1 | cut -d' ' -f2-)
    if [ -z "${CKPT}" ]; then
        echo "ERROR: no model_*.pt found under ${LATEST_RUN}" >&2
        exit 1
    fi
fi

echo "[render_walking] using checkpoint: ${CKPT}"

cd "${MJLAB_DIR}"
PYTHONPATH="${REPO_ROOT}" MUJOCO_GL=egl python -u -m tools.mjlab_integration.g1_walk.run_pretrained \
    --task Mjlab-Velocity-Flat-Unitree-G1 \
    --checkpoint "${CKPT}" \
    --steps 1000 \
    --num-envs 1 \
    --height 720 \
    --width 1280 \
    --out "${OUT_DIR}"

echo "[render_walking] done. mp4 at: ${OUT_DIR}/g1_dance.mp4"
echo "  (renamed from g1_dance to g1_walking for clarity in the report)"
mv "${OUT_DIR}/g1_dance.mp4" "${OUT_DIR}/g1_walking.mp4" 2>/dev/null || true
ls -la "${OUT_DIR}"/*.mp4 2>&1 || true
