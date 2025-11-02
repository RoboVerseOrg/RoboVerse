#!/bin/bash
# OpenVLA LoRA fine-tuning on RoboVerse dataset

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Config
VLA_PATH="$PROJECT_ROOT/third_party/openvla/openvla-7b"
HF_REPO="openvla/openvla-7b"
DATA_ROOT_DIR="$HOME/tensorflow_datasets"
DATASET_NAME="bridge_orig"
RUN_ROOT_DIR="$SCRIPT_DIR/runs"
ADAPTER_TMP_DIR="$SCRIPT_DIR/adapters"
LORA_RANK=32
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-4
IMAGE_AUG="False"
# WandB Configuration
# Set USE_WANDB=false to skip WandB logging (no login required)
# Set USE_WANDB=true and login with: wandb login (or set WANDB_API_KEY env var)
USE_WANDB=false
WANDB_PROJECT="openvla_roboverse"
WANDB_ENTITY=""
SAVE_STEPS=200
MAX_STEPS=200

mkdir -p "$RUN_ROOT_DIR" "$ADAPTER_TMP_DIR"

echo "OpenVLA LoRA Fine-tuning"
echo "Dataset: $DATASET_NAME | BS: $BATCH_SIZE | LR: $LEARNING_RATE | Steps: $MAX_STEPS"

# Download weights if needed
if [ ! -d "$VLA_PATH" ] || [ -z "$(ls -A "$VLA_PATH")" ]; then
  echo "Downloading $HF_REPO..."
  python3 - <<EOF
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
processor = AutoProcessor.from_pretrained("$HF_REPO", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "$HF_REPO",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
processor.save_pretrained("$VLA_PATH")
model.save_pretrained("$VLA_PATH")
print("✓ Downloaded to $VLA_PATH")
EOF
else
  echo "✓ Found weights at $VLA_PATH"
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/roboverse_learn/vla/rlds_utils

# Disable WandB if requested
if [ "$USE_WANDB" = false ]; then
  export WANDB_MODE=disabled
  echo "WandB logging disabled"
else
  echo "WandB logging enabled (project: $WANDB_PROJECT)"
  echo "Note: Make sure you're logged in: wandb login (or set WANDB_API_KEY)"
fi

cd "$PROJECT_ROOT/third_party/openvla"

python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --max_steps "$MAX_STEPS" \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$DATA_ROOT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --adapter_tmp_dir "$ADAPTER_TMP_DIR" \
  --lora_rank "$LORA_RANK" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --image_aug "$IMAGE_AUG" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --save_steps "$SAVE_STEPS"

echo "✓ Fine-tuning complete!"
echo "  Checkpoints: $RUN_ROOT_DIR"
echo "  Adapters: $ADAPTER_TMP_DIR"
