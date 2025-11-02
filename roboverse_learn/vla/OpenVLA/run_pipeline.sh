#!/bin/bash
# RLDS conversion + OpenVLA fine-tuning pipeline
# Prerequisites: Demos must be collected first (see README)
# Usage: bash run_pipeline.sh [--skip-convert]

set -e

TASK_NAME="pick_butter"
SIM_BACKEND="mujoco"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEMO_DIR="$PROJECT_ROOT/roboverse_demo/demo_${SIM_BACKEND}/${TASK_NAME}-"

SKIP_CONVERT=false
echo "$DEMO_DIR"
for arg in "$@"; do
    [ "$arg" = "--skip-convert" ] && SKIP_CONVERT=true
done

echo "=========================================="
echo "OpenVLA Training Pipeline"
echo "Task: $TASK_NAME | Sim: $SIM_BACKEND"
echo "=========================================="

# Step 1: Convert to RLDS
if [ "$SKIP_CONVERT" = false ]; then
    echo "[1/2] Converting to RLDS..."
    cd "$SCRIPT_DIR/../rlds_utils/roboverse"
    mkdir -p demo
    [ -L "demo/${TASK_NAME}-" ] && rm "demo/${TASK_NAME}-"

    if [ -d "$DEMO_DIR" ]; then
        ln -s "$DEMO_DIR" "demo/${TASK_NAME}-"
        echo "✓ Symlink: demo/${TASK_NAME}- -> $DEMO_DIR"
    else
        echo "Error: Demo directory not found: $DEMO_DIR"
        echo "Please collect demos first (see README)"
        exit 1
    fi

    eval "$(conda shell.bash hook)"
    conda activate rlds_env
    tfds build --overwrite
    echo "✓ RLDS conversion done → ~/tensorflow_datasets/bridge_orig/"
else
    echo "[1/2] Skipped (--skip-convert)"
fi

# Step 2: Fine-tune
echo "[2/2] Fine-tuning OpenVLA..."
cd "$SCRIPT_DIR"
eval "$(conda shell.bash hook)"
conda activate openvla

[ -z "$HF_TOKEN" ] && echo "Note: Set HF_TOKEN if downloading weights"

bash finetune.sh

echo ""
echo "✓ Pipeline Complete!"
echo "  Dataset: ~/tensorflow_datasets/bridge_orig/"
echo "  Checkpoints: ./runs/"
echo "  Adapters: ./adapters/"
echo ""
echo "Evaluate: python eval.py --model_path runs/<checkpoint> --task $TASK_NAME"
echo ""
