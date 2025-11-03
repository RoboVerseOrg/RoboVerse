# OpenVLA

OpenVLA is the first open-source 7B Vision-Language-Action model, which was built upon Prismatic VLM. RLDS is the mainstream data format for VLAs. To finetune RoboVerse data on OpenVLA, you need to convert the RoboVerse data to RLDS format. 

We provide automated scripts to streamline the entire training pipeline. This guide covers both the **Quick Start** approach using automated scripts and detailed **Manual Steps** for advanced users.

## Quick Start (Automated Pipeline)

### 1. Setup (one-time)

```bash
cd roboverse_learn/vla/OpenVLA
bash setup_env.sh
```

This creates `rlds_env` and `openvla` conda environments with all dependencies.

### 2. Collect Demos

```bash
cd ../../../../  # Go to RoboVerse root
python scripts/advanced/collect_demo.py --sim=mujoco --task=pick_butter --headless --run_all
```

**Output:** `roboverse_demo/demo_mujoco/pick_butter-/`

### 3. Train

```bash
cd roboverse_learn/vla/OpenVLA

# Convert to RLDS + fine-tune
bash run_pipeline.sh

# Skip conversion if dataset already exists
bash run_pipeline.sh --skip-convert
```

**Output:** 
- `runs/` - Model checkpoints
- `adapters/` - LoRA adapter weights

### 4. Evaluate

```bash
conda activate openvla
python eval.py --model_path runs/<checkpoint> --task pick_butter
```

### Configuration

Edit `finetune.sh` to customize training parameters:

```bash
LORA_RANK=32
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=5000
CUDA_VISIBLE_DEVICES=0
```

**Notes:**
- Set `HF_TOKEN` if downloading OpenVLA weights: `export HF_TOKEN=your_token`
- Change `TASK_NAME` in `run_pipeline.sh` for different tasks
- **WandB**: Login with `wandb login` to enable logging, or set `USE_WANDB=false` in `finetune.sh` to disable

---

## Manual Pipeline (Detailed Steps)

For users who need fine-grained control or want to understand each step, the pipeline consists of four stages:

**Demo Collection → RLDS Conversion → VLA Fine-tuning → Evaluation**

### 1. Collect Demonstration Trajectories

Generate robotic demonstration data using RoboVerse simulation. Example:

```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco \
  --task=pick_butter \
  --headless \
  --run_all
```

**Parameters:**

- `--sim`: Simulation backend (mujoco, isaac, …)
- `--task`: Specific manipulation task (e.g., pick_butter)
- `--headless`: Run without GUI for efficiency
- `--run_all`: Run all available episodes

**Output example:**

```
roboverse_demo/demo_mujoco/pick_butter-/robot-franka/demo_0001
```

### 2. Data Format Conversion (to RLDS)

Collected demos must be converted into the Robot Learning Dataset Specification (RLDS) format.

#### Step 2.1 Navigate to RLDS utilities

```bash
cd roboverse_learn/vla/rlds_utils/roboverse
```

#### Step 2.2 Create demo symlink

The RLDS builder expects data under:

```
roboverse_learn/vla/rlds_utils/roboverse/demo/pick_butter-*
```

If your raw data is under `roboverse_demo/demo_mujoco/pick_butter-`, create a symlink:

```bash
# Ensure demo directory exists
mkdir -p demo

# Link collected data
ln -s /absolute/path/to/roboverse_demo/demo_mujoco/pick_butter- demo/pick_butter-
```

After this, the path should look like:

```
roboverse_learn/vla/rlds_utils/roboverse/demo/pick_butter-/robot-franka/demo_0001
```

#### Step 2.3 Set up conversion environment

```bash
conda env create -f ../environment_ubuntu.yml
conda activate rlds_env
```

**Key packages:** tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam, matplotlib, plotly, wandb.

#### Step 2.4 Run conversion

```bash
tfds build --overwrite
```

**Output dataset:**

```
~/tensorflow_datasets/roboverse_dataset/
```

### 3. Fine-tune OpenVLA

#### Step 3.1 Environment Setup

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch (adjust version for your CUDA setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

#### Step 3.2 Install OpenVLA

```bash
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2
pip install packaging ninja
ninja --version   # should print a version number
pip install "flash-attn==2.5.5" --no-build-isolation
```

**Troubleshooting Flash Attention Installation:**

If you encounter the following error with CUDA 12+ and PyTorch 2.6:

```
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Solution:** Download the appropriate wheel file for your CUDA and PyTorch versions from the official Flash Attention releases: [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

For a stable CUDA 12 environment, we recommend using **flash-attn 2.7.4**.

#### Step 3.3 Fine-tuning Execution
If you haven't downloaded the OpenVLA checkpoints yet, set your Hugging Face token as an environment variable and then run:
```bash
cd roboverse_learn/vla/
export HF_token=your_hf_token
```
Launch training with the RoboVerse wrapper script:

```bash
cd roboverse_learn/vla/
bash finetune_roboverse.sh
```

⚠️ **Make sure dataset paths and model checkpoint paths in finetune_roboverse.sh are correct.**

### 4. Evaluation

After training, evaluate the VLA model on RoboVerse tasks:

```bash
python roboverse_learn/vla/vla_eval.py \
  --model_path openvla_runs/path/to/checkpoint \
  --task pick_butter
```

### 5. Data Pipeline Overview

```
Raw Demos ──collect_demo.py──▶ demo_mujoco/
      │
      ▼
 RLDS Conversion ──tfds──▶ ~/tensorflow_datasets/roboverse_dataset/
      │
      ▼
 OpenVLA Fine-tuning ──finetune_roboverse.sh──▶ Trained VLA
      │
      ▼
 Evaluation ──vla_eval.py──▶ Task Performance Metrics
```