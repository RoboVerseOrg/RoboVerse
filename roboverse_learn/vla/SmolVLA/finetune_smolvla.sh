#!/bin/bash

# Fine-tuning script for SmolVLA on RoboVerse dataset
# Supports single-GPU, multi-GPU, and multi-node training via Hugging Face Accelerate
#
# Usage:
#   Single GPU:   bash finetune_smolvla.sh
#   Multi-GPU:    accelerate launch --num_processes=<N> finetune_smolvla.sh
#   Multi-node:   Use accelerate config first, then accelerate launch

set -e

# ============ Configuration ============
# Dataset configuration
DATASET_REPO_ID="<your_hf_name>/roboverse-pick_butter"  # Change this to your dataset repo
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# Model configuration
PRETRAINED_MODEL="lerobot/smolvla"  # Base SmolVLA model from HuggingFace
OUTPUT_DIR="./smolvla_runs"
RUN_NAME="smolvla_roboverse_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters (aligned with official SmolVLA settings)
BATCH_SIZE=8                # Per-device batch size
LEARNING_RATE=1e-4          # Peak learning rate
TRAINING_STEPS=100000       # Total training steps
WARMUP_STEPS=1000           # Learning rate warmup steps
SAVE_STEPS=5000             # Checkpoint save frequency (official default)
EVAL_STEPS=5000             # Evaluation frequency
DECAY_LR=1e-5               # Final learning rate after cosine decay

# Distributed training configuration
# For multi-GPU training, these will be set automatically by accelerate
NUM_WORKERS=4               # DataLoader workers per process
GRADIENT_ACCUMULATION_STEPS=1  # Gradient accumulation steps

# Device configuration
DEVICE="cuda"  # LeRobot accepts "cuda", "mps", or "cpu"

# Logging
WANDB_PROJECT="smolvla_roboverse"
WANDB_ENTITY=""  # Set your wandb entity if needed
USE_WANDB=true

# ============ Setup ============
echo "========================================="
echo "SmolVLA Fine-tuning on RoboVerse Dataset"
echo "========================================="
echo "Dataset: $DATASET_REPO_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo "Batch size (per device): $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE -> $DECAY_LR"
echo "Training steps: $TRAINING_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if dataset exists
if [ ! -d "$LEROBOT_HOME/$DATASET_REPO_ID" ]; then
    echo "Error: Dataset not found at $LEROBOT_HOME/$DATASET_REPO_ID"
    echo "Please run the data conversion script first:"
    echo "  python roboverse_learn/vla/SmolVLA/convert_roboverse_to_lerobot.py \\"
    echo "    --input-root <path_to_demos> \\"
    echo "    --repo-id $DATASET_REPO_ID \\"
    echo "    --overwrite"
    exit 1
fi

# ============ Distributed Training Setup ============
# Check if running under accelerate (multi-GPU/multi-node)
if [ -z "$ACCELERATE_TORCH_DEVICE" ]; then
    echo "Running in single-GPU mode"
    echo "For multi-GPU training, use: accelerate launch --num_processes=N $0"
    # Single GPU: Set visible devices
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    echo "Running under accelerate (distributed mode)"
    echo "Accelerate will handle device assignment"
fi

# ============ Training ============
echo "Starting fine-tuning..."

# Build training command using lerobot-train
# Note: LeRobot uses Hugging Face Accelerate for distributed training
# All distributed configurations are handled automatically by accelerate

TRAIN_CMD="lerobot-train \
    --policy.type smolvla \
    --dataset.repo_id $DATASET_REPO_ID \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --batch_size $BATCH_SIZE \
    --grad_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --optimizer.type adamw \
    --optimizer.lr $LEARNING_RATE \
    --optimizer.weight_decay 0.0 \
    --optimizer.grad_clip_norm 1.0 \
    --steps $TRAINING_STEPS \
    --scheduler.type cosine_decay_with_warmup \
    --scheduler.num_warmup_steps $WARMUP_STEPS \
    --scheduler.num_decay_steps $TRAINING_STEPS \
    --scheduler.peak_lr $LEARNING_RATE \
    --scheduler.decay_lr $DECAY_LR \
    --eval_freq $EVAL_STEPS \
    --log_freq 100 \
    --save_freq $SAVE_STEPS \
    --num_workers $NUM_WORKERS \
    --policy.device $DEVICE \
    --policy.use_amp true \
    --policy.push_to_hub false"

# Add wandb if enabled
if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD \
        --wandb.enable true \
        --wandb.project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb.entity $WANDB_ENTITY"
    fi
else
    TRAIN_CMD="$TRAIN_CMD --wandb.enable false"
fi

# Execute training
echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD

echo ""
echo "========================================="
echo "Fine-tuning completed!"
echo "Checkpoints saved in: $OUTPUT_DIR/$RUN_NAME"
echo "========================================="

# ============ Save training info ============
cat > "$OUTPUT_DIR/$RUN_NAME/training_info.txt" <<EOF
SmolVLA Fine-tuning on RoboVerse
=================================
Date: $(date)
Dataset: $DATASET_REPO_ID
Base model: $PRETRAINED_MODEL

Training Configuration:
- Batch size (per device): $BATCH_SIZE
- Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS
- Learning rate: $LEARNING_RATE (peak) -> $DECAY_LR (final)
- Training steps: $TRAINING_STEPS
- Warmup steps: $WARMUP_STEPS
- Save frequency: $SAVE_STEPS steps
- Evaluation frequency: $EVAL_STEPS steps

Distributed Training:
- Framework: Hugging Face Accelerate
- Mixed precision: Enabled (AMP)
- Gradient clipping: 1.0

Output: $OUTPUT_DIR/$RUN_NAME
EOF

echo "Training info saved to: $OUTPUT_DIR/$RUN_NAME/training_info.txt"

# ============ Multi-GPU Training Instructions ============
cat <<'EOF_INSTRUCTIONS'

Multi-GPU Training Instructions:
=================================

1. Single Machine, Single GPU (default):
   bash finetune_smolvla.sh

2. Single Machine, Multiple GPUs:
   accelerate launch --num_processes=<NUM_GPUS> finetune_smolvla.sh

   Example for 4 GPUs:
   accelerate launch --num_processes=4 finetune_smolvla.sh

3. Multiple Machines (Multi-node):
   a) First, configure accelerate on each node:
      accelerate config

   b) Then launch on each node:
      accelerate launch --num_machines=<N> \
                       --machine_rank=<RANK> \
                       --main_process_ip=<MAIN_IP> \
                       --main_process_port=<PORT> \
                       finetune_smolvla.sh

For more details, see: https://huggingface.co/docs/accelerate/

EOF_INSTRUCTIONS
