# OpenVLA Fine-tuning for RoboVerse

Automated scripts for fine-tuning OpenVLA models on RoboVerse demonstration data.

## Quick Start

### 1. Setup (one-time)
```bash
bash setup_env.sh
```

### 2. Collect Demos
```bash
cd ../../../../  # Go to RoboVerse root
python scripts/advanced/collect_demo.py --sim=mujoco --task=pick_butter --headless --run_all
```

### 3. Train
```bash
cd roboverse_learn/vla/OpenVLA

# Convert to RLDS + fine-tune
bash run_pipeline.sh

# Skip conversion if dataset already exists
bash run_pipeline.sh --skip-convert
```

### 4. Evaluate
```bash
conda activate openvla
python eval.py --model_path runs/<checkpoint> --task pick_butter
```

## Scripts

**setup_env.sh** - Sets up `rlds_env` and `openvla` conda environments

**run_pipeline.sh** - RLDS conversion + fine-tuning for `pick_butter` task (edit TASK_NAME to change)

**finetune.sh** - OpenVLA LoRA fine-tuning (rank=32, bs=8, lr=5e-4, steps=5000)

**eval.py** - Model evaluation script

## Demo Collection

Collect demos separately before running the pipeline:
```bash
cd RoboVerse/
python scripts/advanced/collect_demo.py --sim=mujoco --task=pick_butter --headless --run_all
```

Output: `roboverse_demo/demo_mujoco/pick_butter-/`

## Outputs

- `runs/` - Model checkpoints
- `adapters/` - LoRA adapter weights

## Configuration

Edit `finetune.sh` to customize:
```bash
LORA_RANK=32
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=5000
```

## Notes

- Set `HF_TOKEN` if downloading OpenVLA weights: `export HF_TOKEN=your_token`
- Change GPU in `finetune.sh`: `CUDA_VISIBLE_DEVICES=0`
- Dataset path: Modify `DATA_ROOT_DIR` if needed
- **WandB**: To enable logging, login with `wandb login` (or set `WANDB_API_KEY` env var). 
  To skip WandB logging, set `USE_WANDB=false` in `finetune.sh`

