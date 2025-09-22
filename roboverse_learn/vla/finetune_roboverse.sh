#!/bin/bash

# Fine-tuning script for roboverse_dataset using OpenVLA LoRA
# Usage: bash finetune_roboverse.sh

set -e

# Configuration
VLA_PATH="third_party/openvla/openvla-7b"
DATA_ROOT_DIR="."
DATASET_NAME="roboverse_dataset"
RUN_ROOT_DIR="./openvla_runs"
ADAPTER_TMP_DIR="./openvla_adapters"
LORA_RANK=32
BATCH_SIZE=8  # Reduced for smaller dataset
GRAD_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-4
IMAGE_AUG="False"
WANDB_PROJECT="openvla_roboverse"
WANDB_ENTITY=""  # Change this to your wandb entity
SAVE_STEPS=5000
MAX_STEPS=5000
# Create directories if they don't exist
mkdir -p "$RUN_ROOT_DIR"
mkdir -p "$ADAPTER_TMP_DIR"

echo "Starting OpenVLA LoRA fine-tuning on roboverse_dataset..."
echo "Data root: $DATA_ROOT_DIR"
echo "Dataset: $DATASET_NAME"
echo "Run directory: $RUN_ROOT_DIR"
echo "Adapter directory: $ADAPTER_TMP_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUMULATION_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Image augmentation: $IMAGE_AUG"

# Set GPU device (change 0 to your desired GPU ID)
export CUDA_VISIBLE_DEVICES=5 # Use GPU 1 instead of GPU 0

# Launch the fine-tuning script
cd third_party/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
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
  --save_steps "$SAVE_STEPS" \


echo "Fine-tuning completed!"
echo "Checkpoints saved in: $RUN_ROOT_DIR"
echo "Adapters saved in: $ADAPTER_TMP_DIR"
