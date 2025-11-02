# SmolVLA

SmolVLA is a lightweight Vision-Language-Action (VLA) model from Hugging Face's LeRobot framework. It provides a faster and more efficient alternative to larger VLA models like OpenVLA, making it ideal for rapid experimentation and resource-constrained environments.

## Overview

SmolVLA uses the same LeRobot data format as the pi0 models, enabling data reuse between these implementations. This guide covers the complete pipeline from data conversion to model evaluation.

## Pipeline Overview

The workflow consists of three main stages:

**Demo Collection → LeRobot Conversion → SmolVLA Training → Evaluation**

### 1. Collect Demonstration Trajectories

Generate robotic demonstration data using RoboVerse simulation:

```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco \
  --task=pick_butter \
  --headless \
  --run_all
```

**Output example:**
```
roboverse_demo/demo_mujoco/pick_butter-/robot-franka/demo_0001
```

### 2. Data Format Conversion (to LeRobot)

Convert collected demos to LeRobot format using the shared conversion script:

```bash
python roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root roboverse_demo/demo_mujoco \
  --repo-id <your_hf_name>/roboverse-pick_butter \
  --overwrite
```

**Parameters:**
- `--input-root`: Directory containing demonstration data
- `--repo-id`: Hugging Face repository ID for the dataset
- `--overwrite`: Overwrite existing dataset if present

**Output:** Dataset saved to `~/.cache/huggingface/lerobot/<repo-id>`

### 3. Install SmolVLA and LeRobot

Install LeRobot with SmolVLA support:

```bash
cd third_party
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

### 4. Configure Training

Edit `roboverse_learn/vla/SmolVLA/finetune_smolvla.sh` to set your dataset:

```bash
DATASET_REPO_ID="<your_hf_name>/roboverse-pick_butter"
```

**Key hyperparameters:**
- `BATCH_SIZE`: Batch size for training (default: 8)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `TRAINING_STEPS`: Total training steps (default: 100000)
- `WARMUP_STEPS`: Learning rate warmup steps (default: 1000)
- `SAVE_STEPS`: Checkpoint save frequency (default: 50)

### 5. Train SmolVLA

Run the training script:

```bash
cd roboverse_learn/vla/SmolVLA
bash finetune_smolvla.sh
```

**Training output:** Checkpoints saved to `./smolvla_runs/smolvla_roboverse_<timestamp>/checkpoints/`

### 6. Evaluate Model

Evaluate the trained model on RoboVerse tasks:

```bash
python roboverse_learn/vla/SmolVLA/smolvla_eval.py \
  --model_path ./smolvla_runs/smolvla_roboverse_<timestamp>/checkpoints/001000 \
  --task pick_butter \
  --robot franka \
  --sim mujoco \
  --num_episodes 10
```

**Parameters:**
- `--model_path`: Path to trained checkpoint
- `--task`: Task name (e.g., pick_butter)
- `--robot`: Robot model (e.g., franka)
- `--sim`: Simulator backend (mujoco, pybullet, sapien, etc.)
- `--num_episodes`: Number of evaluation episodes
- `--max_steps`: Maximum steps per episode (default: 250)

**Output:** 
- Videos: `./smolvla_eval_output/episode_*.mp4`
- Results: `./smolvla_eval_output/smolvla_eval_<task>_<timestamp>.json`

## Data Pipeline Diagram

```
Raw Demos ──collect_demo.py──▶ demo_mujoco/
      │
      ▼
LeRobot Conversion ──convert_roboverse_to_lerobot.py──▶ ~/.cache/huggingface/lerobot/
      │
      ▼
SmolVLA Training ──finetune_smolvla.sh──▶ Trained Model
      │
      ▼
Evaluation ──smolvla_eval.py──▶ Performance Metrics & Videos
```

## Model Comparison

| Feature | OpenVLA | SmolVLA | pi0 |
|---------|---------|---------|-----|
| Model Size | ~7B parameters | <1B parameters | Various sizes |
| Data Format | RLDS | LeRobot | LeRobot |
| Training Speed | Slow | Fast | Medium |
| Inference Speed | Slow | Fast | Medium |
| Resource Requirements | High | Low | Medium |
| Best Use Case | Research accuracy | Fast experimentation | Production deployment |

## Key Advantages

1. **Data Sharing**: SmolVLA uses the same LeRobot format as pi0, so you only need to convert data once
2. **Lightweight**: Smaller model size enables faster training and inference
3. **Lower Resources**: Can run on GPUs with less memory
4. **Quick Iteration**: Faster training cycles for experimentation

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `finetune_smolvla.sh`:
```bash
BATCH_SIZE=4
```

### Dataset Not Found

Check that the dataset was converted successfully:
```bash
ls ~/.cache/huggingface/lerobot/
```

### Import Errors

Ensure LeRobot is installed with SmolVLA support:
```bash
pip install -e ".[smolvla]"
```

### Model Loading Fails

Verify the checkpoint path contains the `pretrained_model/` subdirectory:
```bash
ls <checkpoint_path>/pretrained_model/
```

## Advanced Usage

### Multi-task Training

Convert multiple tasks to the same dataset repository:
```bash
python roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root roboverse_demo/demo_mujoco/pick_butter- \
  --repo-id myname/roboverse-multitask

python roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root roboverse_demo/demo_mujoco/stack_blocks- \
  --repo-id myname/roboverse-multitask
```

### Custom Hyperparameters

Modify `finetune_smolvla.sh` to adjust training parameters:
- Learning rate schedule: Modify `DECAY_LR` for final learning rate
- Gradient clipping: Change `--optimizer.grad_clip_norm`
- Logging frequency: Adjust `--log_freq`

### WandB Logging

Enable Weights & Biases logging in `finetune_smolvla.sh`:
```bash
USE_WANDB=true
WANDB_PROJECT="smolvla_roboverse"
WANDB_ENTITY="<your_wandb_username>"
```

## File Structure

```
roboverse_learn/vla/SmolVLA/
├── README.md                    # Detailed usage documentation
├── finetune_smolvla.sh         # Training script
├── smolvla_eval.py             # Evaluation script
└── roboverse_smolvla_policy.py # Data format adapter (optional)
```

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLA on Hugging Face](https://huggingface.co/lerobot/smolvla)
- [OpenVLA Documentation](./openvla.md)
- [pi0 Documentation](../../vla/pi0/README.md)

## Related Documentation

- Main VLA README: `roboverse_learn/vla/README.md`
- Data conversion: `roboverse_learn/vla/SmolVLA/convert_roboverse_to_lerobot.py`
- pi0 implementation: `roboverse_learn/vla/pi0/README.md`

