# RoboVerse VLA Training Pipeline

A comprehensive workflow for training Vision-Language-Action (VLA) models using RoboVerse robotic manipulation data.

## Quick Start (Automated)

### One-time Setup
```bash
# Install both RLDS and OpenVLA environments automatically
cd roboverse_learn/vla/OpenVLA
bash setup_env.sh
```

### Collect Demonstration Data
```bash
# Collect demos first (run from RoboVerse root)
python scripts/advanced/collect_demo.py --sim=mujoco --task=pick_butter --headless --run_all
```

### Train Model
```bash
cd roboverse_learn/vla/OpenVLA

# Convert to RLDS and fine-tune
bash run_pipeline.sh

# Skip RLDS conversion if dataset already exists
bash run_pipeline.sh --skip-convert
```

### Evaluate Trained Model
```bash
cd roboverse_learn/vla/OpenVLA
conda activate openvla
python eval.py --model_path runs/<checkpoint> --task pick_butter
```

## Supported VLA Models

This folder contains implementations for multiple VLA architectures:

- **OpenVLA** - Full-size VLA model using RLDS data format (see main sections below)
- **SmolVLA** - Lightweight VLA from Hugging Face/LeRobot (see `SmolVLA/README.md`)
- **π0 family** - Physical Intelligence's π models (see `pi0/README.md`)

Each model has its own subdirectory with specific training and evaluation scripts.



## OpenVLA Scripts

All OpenVLA automation scripts are in the `OpenVLA/` directory. See [`OpenVLA/README.md`](OpenVLA/README.md) for details.

**OpenVLA/setup_env.sh** - Sets up `rlds_env` (data conversion) and `openvla` (training/eval) environments

**OpenVLA/run_pipeline.sh** - RLDS conversion + fine-tuning for `pick_butter` task
  - Prerequisite: Collect demos first with `collect_demo.py`
  - Args: `--skip-convert`

**OpenVLA/finetune.sh** - LoRA fine-tuning with configurable hyperparameters

**OpenVLA/eval.py** - Model evaluation script

## Workflow (Manual Steps)

### Step 1: Collect Demonstration Trajectories

Generate robotic demonstration data using the RoboVerse simulation environment:

```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco --task=pick_butter --headless --run_all
```

**Parameters:**
- `--sim`: Simulation backend (mujoco)
- `--task`: Specific manipulation task (pick_butter)
- `--headless`: Run without GUI for efficiency
- `--run_all`: Execute all available episodes

### Step 2: Data Format Conversion to RLDS

Convert collected demonstrations to the Robot Learning Dataset Specification (RLDS) format:

```bash
# Navigate to RLDS utilities directory
cd roboverse_learn/vla/rlds_utils/roboverse/

# Create symbolic link for demo data
# Note: Adjust folder name according to your task & simulation setup
mkdir -p demo
ln -s /absolute/path/to/roboverse_demo/demo_mujoco/pick_butter- demo/pick_butter-



# Set up conversion environment
conda env create -f ../environment_ubuntu.yml
conda activate rlds_env

# Convert to RLDS format
tfds build --overwrite
```

**Output:** Converted dataset stored in `~/tensorflow_datasets/roboverse_dataset/`

### Step 3: OpenVLA Model Fine-tuning

#### 3.1 Environment Setup

```bash
# Navigate to third-party dependencies
cd thirdparty

# Clone OpenVLA repository
git clone https://github.com/openvla/openvla.git
```

#### 3.2 Installation

Follow the official OpenVLA installation guide:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch with CUDA support
# Check https://pytorch.org/get-started/locally/ for platform-specific instructions
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install OpenVLA package
cd openvla
pip install -e .

# Install Flash Attention 2 for efficient training
# Reference: https://github.com/Dao-AILab/flash-attention
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja installation (should return exit code "0")
pip install "flash-attn==2.5.5" --no-build-isolation
```

#### 3.3 Fine-tuning Execution
If you haven't downloaded the OpenVLA checkpoints yet, set your Hugging Face token as an environment variable and then run:
```bash
cd roboverse_learn/vla/
export HF_token=your_hf_token
bash finetune_roboverse.sh
```

**Important:** Ensure dataset and model paths in the script match your actual directory structure.

### Step 4: Model Evaluation

Assess the trained model's performance on specific tasks:

```bash
python roboverse_learn/vla/vla_eval.py \
  --model_path openvla_runs/path/to/checkpoint \
  --task pick_butter
```

## Data Pipeline

```
Demo Collection → RLDS Conversion → VLA Fine-tuning → Model Evaluation
     ↓                    ↓                ↓                ↓
Raw Trajectories → Standardized Format → Trained Model → Performance Metrics
```

## Dataset Information

- **Format:** RLDS (Robot Learning Dataset Specification)
- **Storage Location:** `~/tensorflow_datasets/roboverse_dataset/`

## Configuration Notes

- Adjust simulation parameters in `collect_demo.py` based on your specific tasks
- Modify dataset paths in `finetune_roboverse.sh` to match your environment

## Quick Start for Different Models

### OpenVLA (this workflow)
```bash
cd OpenVLA
bash run_pipeline.sh
python eval.py --model_path runs/<checkpoint> --task pick_butter
```

### SmolVLA (lightweight alternative)
```bash
# See SmolVLA/README.md for details
cd SmolVLA
bash finetune_smolvla.sh
python smolvla_eval.py --model_path <checkpoint> --task pick_butter
```

### π0 Models (Physical Intelligence)
```bash
# See pi0/README.md for details
cd pi0
python convert_roboverse_to_lerobot.py --input-root <demos> --repo-id <dataset>
# Then follow pi0/README.md for training
```

## Choosing a Model

| Model | Size | Data Format | Best For |
|-------|------|-------------|----------|
| **OpenVLA** | ~7B params | RLDS | High accuracy, research |
| **SmolVLA** | <1B params | LeRobot | Fast training/inference, resource-constrained |
| **π0** | Various | LeRobot | Production deployment, multi-task |

## References

- [OpenVLA Repository](https://github.com/openvla/openvla)
- [LeRobot & SmolVLA](https://github.com/huggingface/lerobot)
- [Physical Intelligence π0](https://github.com/physical-intelligence/openpi)

