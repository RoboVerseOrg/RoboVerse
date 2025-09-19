# RoboVerse VLA Training Pipeline

Complete Vision-Language-Action model training workflow

## 🚀 Training Steps

### 1. Collect Demo Trajectories
```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco --task=pick_butter --robot=franka \
  --max_demo_idx=100 --headless --run_all
```

### 2. Create Soft Link and Convert to RLDS Format
```bash
# Navigate to rlds utils directory
cd roboverse_learn/vla/rlds_utils/roboverse/

# Create soft link for demo data
ln -s ../../../../roboverse_demo demo

# Activate environment
conda env create -f ../environment_ubuntu.yml
conda activate rlds_env

# Convert to RLDS format
tfds build --overwrite

# Verify conversion results
cd ..
python visualize_dataset.py roboverse_dataset
```

### 3. Fine-tune OpenVLA Model
```bash
cd roboverse_learn/vla/
bash finetune_roboverse.sh
```

### 4. Evaluate Model Performance
```bash
python roboverse_learn/vla/vla_eval.py \
  --model_path openvla_runs/path/to/checkpoint \
  --task pick_butter \
  --num_episodes 10
```

## 📁 Data Flow

```
Collect Demo → Soft Link → RLDS Convert → VLA Finetune → Model Eval
```

Converted dataset is stored in `~/tensorflow_datasets/roboverse_dataset/`