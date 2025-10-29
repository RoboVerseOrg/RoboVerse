# Using SmolVLA on RoboVerse

This guide walks through the end-to-end workflow for training and fine-tuning SmolVLA (Hugging Face's lightweight VLA model from LeRobot) on RoboVerse demonstrations.

## Overview

SmolVLA is a lightweight Vision-Language-Action (VLA) model with <1B parameters, offering faster training and inference compared to larger VLA models while maintaining strong performance on robotic manipulation tasks.

**Key Features:**
- Lightweight architecture (<1B parameters)
- Shared LeRobot data format with pi0 models
- Distributed training support (single-GPU, multi-GPU, multi-node)
- Built-in evaluation with video recording and metrics

For detailed documentation, see the [full SmolVLA guide](../../../docs/source/roboverse_learn/imitation_learning/smolvla.md).

## Workflow

### Step 1: Collect Demonstration Trajectories

Generate robotic demonstration data using the RoboVerse simulation environment:

```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco \
  --task=pick_butter \
  --headless \
  --run_all
```

**Parameters:**
- `--sim`: Simulation backend (mujoco, pybullet, sapien3)
- `--task`: Specific manipulation task (pick_butter, stack_blocks, etc.)
- `--headless`: Run without GUI for efficiency
- `--run_all`: Execute all available episodes

**Output:** Demonstrations saved in `roboverse_demo/demo_mujoco/pick_butter-/`

### Step 2: Install LeRobot with SmolVLA

```bash
cd third_party
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# Also install accelerate for distributed training
pip install accelerate
```

### Step 3: Convert Data to LeRobot Format

Convert collected demonstrations to LeRobot dataset format:

```bash
python roboverse_learn/vla/SmolVLA/convert_roboverse_to_lerobot.py \
  --input-root roboverse_demo/demo_mujoco \
  --repo-id <your_hf_name>/roboverse-pick_butter \
  --overwrite
```

**Note:** This script automatically skips failed demos (missing metadata.json or rgb.mp4 files).

**Output:** Dataset saved to `~/.cache/huggingface/lerobot/<repo_id>`

### Step 4: Configure and Train

Edit `finetune_smolvla.sh` to set your dataset ID:

```bash
# In finetune_smolvla.sh, change:
DATASET_REPO_ID="your_username/roboverse-pick_butter"
```

Then run training:

**Single GPU:**
```bash
cd roboverse_learn/vla/SmolVLA
bash finetune_smolvla.sh
```

**Multi-GPU (Single Machine):**
```bash
cd roboverse_learn/vla/SmolVLA
accelerate launch --num_processes=4 finetune_smolvla.sh  # For 4 GPUs
```

**Multi-Node (Multiple Machines):**
```bash
# First, configure accelerate on each node
accelerate config

# Then launch on each node
accelerate launch --num_machines=<N> \
                 --machine_rank=<RANK> \
                 --main_process_ip=<MAIN_IP> \
                 --main_process_port=<PORT> \
                 finetune_smolvla.sh
```

Training will save checkpoints to `./smolvla_runs/smolvla_roboverse_<timestamp>/checkpoints/`

### Step 5: Evaluate Trained Model

Run evaluation on RoboVerse tasks:

```bash
cd roboverse_learn/vla/SmolVLA
python smolvla_eval.py \
  --model_path ./smolvla_runs/smolvla_roboverse_<timestamp>/checkpoints/005000 \
  --task pick_butter \
  --robot franka \
  --sim mujoco \
  --num_episodes 10 \
  --max_steps 250 \
  --output_dir ./smolvla_eval_output
```

**Output:**
- Episode videos: `smolvla_eval_output/episode_001.mp4`, `episode_002.mp4`, ...
- Evaluation metrics: `smolvla_eval_output/smolvla_eval_pick_butter_<timestamp>.json`

## Files

- `convert_roboverse_to_lerobot.py` - Data conversion script (LeRobot 0.3+ compatible)
- `finetune_smolvla.sh` - Training script with distributed training support
- `smolvla_eval.py` - Evaluation script with video recording and metrics
- `roboverse_smolvla_policy.py` - Optional data adapter for advanced customization

## Training Hyperparameters

SmolVLA training is configured with hyperparameters aligned to the official LeRobot recommendations:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size (per device) | 8 | Adjust based on GPU memory |
| Learning rate | 1e-4 → 1e-5 | Cosine decay with warmup |
| Training steps | 100,000 | Total optimization steps |
| Warmup steps | 1,000 | Linear warmup period |
| Save frequency | 5,000 steps | Checkpoint saving interval |
| Gradient clipping | 1.0 | Norm threshold |
| Optimizer | AdamW | Weight decay = 0.0 |
| Scheduler | Cosine decay | With warmup |
| Mixed precision | Enabled | Automatic mixed precision (AMP) |

These settings can be modified in `finetune_smolvla.sh`.

## Training Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **Single GPU** | `bash finetune_smolvla.sh` | Development, small datasets |
| **Multi-GPU** | `accelerate launch --num_processes=N finetune_smolvla.sh` | Faster training, large batch sizes |
| **Multi-Node** | `accelerate launch --num_machines=M ...` | Very large datasets, maximum speed |

## Data Format

SmolVLA uses the LeRobot dataset format (same as pi0):

```
LeRobot Dataset Structure:
~/.cache/huggingface/lerobot/<repo-id>/
├── episodes/              # Episode data
│   ├── episode_0.parquet
│   ├── episode_1.parquet
│   └── ...
├── images/                # Video frames
│   └── observation.image/
│       ├── episode_0/
│       └── ...
├── meta/                  # Metadata
└── info.json             # Dataset info
```

**Keys in each frame:**
- `observation.image`: RGB image (H, W, 3)
- `observation.state`: Joint positions (9-dim for Franka)
- `action`: Target joint positions (9-dim)
- `task`: Text instruction/task description

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `finetune_smolvla.sh`:
```bash
BATCH_SIZE=4  # Reduce from 8
```
Or increase gradient accumulation:
```bash
GRADIENT_ACCUMULATION_STEPS=2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
```

### Model Loading Fails During Evaluation
Ensure the checkpoint path contains the `pretrained_model/` subdirectory:
```bash
ls <checkpoint_path>/pretrained_model/  # Should show model files
```

### Dataset Not Found
Verify dataset conversion was successful:
```bash
ls ~/.cache/huggingface/lerobot/<repo-id>
```
If missing, re-run the conversion script.

### Multi-GPU Not Working
Check accelerate installation:
```bash
pip install accelerate
accelerate config  # Configure for your setup
accelerate test    # Test configuration
```

### Training Hangs or Crashes
Check CUDA compatibility and ensure all GPUs are accessible:
```bash
nvidia-smi  # Check GPU status
python -c "import torch; print(torch.cuda.device_count())"
```

## Comparison with Other VLA Models

| Feature | SmolVLA | OpenVLA | pi0 |
|---------|---------|---------|-----|
| Model size | <1B params | ~7B params | Various (1B-20B) |
| Data format | LeRobot | RLDS | LeRobot |
| Training speed | Fast | Slow | Medium |
| Inference speed | Fast | Slow | Medium |
| Memory usage | Low | High | Medium |
| Action space | EE control | EE control | Joint control |
| Best for | Fast iteration | High accuracy | Production |

## References

- [Full Documentation](../../../docs/source/roboverse_learn/imitation_learning/smolvla.md)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLA on Hugging Face](https://huggingface.co/lerobot/smolvla)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate/)
- [RoboVerse VLA Overview](../README.md)
