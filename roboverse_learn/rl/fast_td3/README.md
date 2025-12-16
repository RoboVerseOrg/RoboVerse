# FastTD3 Training Configuration

## Configuration Structure

```
configs/
├── base.yaml              # Base config (IsaacGym + H1)
├── mjx_rl_pick.yaml      # MJX pick task
├── mjx_walk.yaml         # MJX + walk
├── mjx_stand.yaml        # MJX + stand  
├── mjx_run.yaml          # MJX + run
├── isaacgym_walk.yaml    # IsaacGym + walk
├── isaacgym_stand.yaml   # IsaacGym + stand
└── isaacgym_run.yaml     # IsaacGym + run
```

## Usage

### Basic Command
```bash
python roboverse_learn/rl/fast_td3/train.py --config <config_name>
```

### Available Configurations
```bash
# MJX Tasks
python roboverse_learn/rl/fast_td3/train.py --config mjx_walk.yaml
python roboverse_learn/rl/fast_td3/train.py --config mjx_stand.yaml
python roboverse_learn/rl/fast_td3/train.py --config mjx_run.yaml
python roboverse_learn/rl/fast_td3/train.py --config mjx_rl_pick.yaml

# IsaacGym Tasks  
python roboverse_learn/rl/fast_td3/train.py --config isaacgym_walk.yaml
python roboverse_learn/rl/fast_td3/train.py --config isaacgym_stand.yaml
python roboverse_learn/rl/fast_td3/train.py --config isaacgym_run.yaml

# Default config
python roboverse_learn/rl/fast_td3/train.py  
```

## Configuration Notes

- **MJX**: Uses Franka robot, suitable for pick tasks
- **IsaacGym**: Uses H1 humanoid robot, suitable for locomotion tasks
- Each config only defines key differences, other params inherit from base.yaml

## Custom Configuration

1. Copy existing config file
2. Modify key parameters (sim, robots, task, etc.)
3. Run: `python roboverse_learn/rl/fast_td3/train.py --config your_config.yaml`

## Checkpoint Saving

To enable checkpoint saving during training, add the following to your YAML config:

```yaml
save_interval: 10000    # Save every 10k steps (set to 0 to disable)
model_dir: "models"     # Directory to save checkpoints (default: "models")
run_name: "my_exp"      # Custom name for saved models (default: task name)
```

Checkpoints will be saved as: `{model_dir}/{run_name}_{global_step}.pt`

## Evaluation

### Basic Evaluation

**By default, evaluation will:**
- ✅ Run 1 episode per environment (collect multiple episodes in parallel)
- ✅ Render and save separate videos for each episode  
- ✅ Save trajectories using handler states (actions + states)
- ✅ Save to `output/eval_rollout_env00_ep00.mp4` and `eval_trajs/*.pkl`

Simple evaluation with all features enabled:

```bash
# Default: 1 episode per env, with rendering and trajectory saving
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt

# Run multiple episodes per environment
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 5

# Disable rendering (faster, trajectory only)
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --render 0

# Disable trajectory saving (video only)
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --save_traj 0
```

### Per-Episode Video Rendering (Default: Enabled)

By default, each episode gets its own video file with performance stats in the log:

```bash
# Default: separate video for each episode
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 5

# This will create:
# - output/eval_rollout_ep000.mp4 (return: 45.23)
# - output/eval_rollout_ep001.mp4 (return: 52.67)
# - output/eval_rollout_ep002.mp4 (return: 48.91)
# - output/eval_rollout_ep003.mp4 (return: 55.34)
# - output/eval_rollout_ep004.mp4 (return: 51.02)

# Save single combined video instead
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 5 \
    --no_render_each_episode \
    --video_path output/eval_combined.mp4
```

### Trajectory Saving (New!)

Save **trajectories** (actions and states) during evaluation for later replay or analysis:

```bash
# Save trajectories with actions only
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 10 \
    --save_traj \
    --traj_dir eval_trajs

# Save trajectories with full states (larger file size)
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 10 \
    --save_traj \
    --save_states \
    --save_every_n_steps 1 \
    --traj_dir eval_trajs

# Combine trajectory saving with video rendering
python roboverse_learn/rl/fast_td3/evaluate.py \
    --checkpoint models/walk_10000.pt \
    --num_episodes 5 \
    --save_traj \
    --save_states \
    --render_each_episode \
    --video_path output/eval.mp4 \
    --traj_dir eval_trajs

# This will create:
# - eval_trajs/walk_h1_eval_20250125_143022_v2.pkl (trajectory file)
# - output/eval_ep000.mp4 ... output/eval_ep004.mp4 (videos)
```

The saved trajectory file can be replayed using the replay scripts in `scripts/advanced/`.

### Evaluation Arguments

**Basic:**
- `--checkpoint PATH`: Path to checkpoint file (default: models/walk_1400.pt)
- `--num_episodes N`: Number of episodes to evaluate (default: 10)
- `--device_rank N`: GPU device rank (default: 0)
- `--num_envs N`: Number of parallel environments (default: from checkpoint config)
- `--headless`: Run in headless mode

**Video Rendering:**
- `--render`: Render and save a single combined video
- `--render_each_episode`: **Save a separate video for each episode** (recommended for analysis)
- `--video_path PATH`: Base path for video(s) (default: output/eval_rollout.mp4)

**Trajectory Saving:**
- `--save_traj`: **Save trajectories during evaluation** (actions and states)
- `--save_states`: Save full states (not just actions) when saving trajectories
- `--save_every_n_steps N`: Save every N steps for downsampling (default: 5, 1=save all)
- `--traj_dir PATH`: Directory to save trajectories (default: eval_trajs)

### Resume Training from Checkpoint

To resume training from a checkpoint, add to your config:

```yaml
checkpoint_path: "models/my_exp_10000.pt"
```

Then run training as usual:
```bash
python roboverse_learn/rl/fast_td3/train.py --config your_config.yaml
```

## V-HACD (Volumetric Hierarchical Approximate Convex Decomposition)

Add the following code in RoboVerse/metasim/sim/isaacgym/isaacgym.py


```python
# eg RigidObjCfg after line 395
asset_options.vhacd_enabled = True 
asset_options.vhacd_params.resolution = 500000  # 
asset_options.vhacd_params.max_convex_hulls = 64 # 
asset_options.vhacd_params.max_num_vertices_per_ch = 64
asset_options.thickness = 0.001

```



## Reconstruct Local Offset from World Coordinates

This script implements calculate the local offset between the desired grasp position and the object center.

### Logic Overview
* **Quaternion Format**: `[w, x, y, z]`

## Implementation Code

```python
import torch

def calculate_local_offset(root_pos, root_rot, target_pos):
    """ 
    Inverse: Calculate local offset of Target in Root coordinate system
    [W, X, Y, Z] format version
    """
    # 1. Calculate world coordinate difference
    diff_world = target_pos - root_pos
    
    # 2. Extract quaternion (format: w, x, y, z)
    # Correction: Index 0 is w
    w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
    
    # 3. Construct inverse rotation (conjugate quaternion: [w, -x, -y, -z])
    # w remains unchanged, vector part is negated
    q_vec_inv = torch.stack([-x, -y, -z], dim=1) 
    w = w.unsqueeze(1)
    
    # 4. Apply rotation: q_inv * diff * q
    # t = 2 * cross(q_vec, v)
    t = 2.0 * torch.cross(q_vec_inv, diff_world, dim=1)
    
    # result = v + w*t + cross(q_vec, t)
    local_offset = diff_world + w * t + torch.cross(q_vec_inv, t, dim=1)
    
    return local_offset

def get_world_pos(root_pos, root_rot, local_offset):
    """ 
    Forward verification: Calculate world coordinates based on local offset
    [W, X, Y, Z] format version
    """
    # 1. Extract quaternion (format: w, x, y, z)
    # Correction: Index 0 is w
    w, x, y, z = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
    
    # 2. Forward rotation q (w, x, y, z)
    # Vector part uses x, y, z directly
    q_vec = torch.stack([x, y, z], dim=1) 
    w = w.unsqueeze(1)
    
    # Input vector v
    v = local_offset
    
    # 3. Apply rotation: q * v * q_inv
    t = 2.0 * torch.cross(q_vec, v, dim=1)
    final_vec = v + w * t + torch.cross(q_vec, t, dim=1)
    
    return root_pos + final_vec

# --- Example Usage & Verification ---

if __name__ == "__main__":
    # Root Position (e.g., Knife Handle)
    knife_pos = torch.tensor([0.201373, -0.330642, 0.779824]).unsqueeze(0)

    # Root Rotation (Quaternion [w, x, y, z])
    # Assuming the data [-0.398, ...] corresponds to wxyz
    knife_rot = torch.tensor([-0.398238, 0.035423, -0.027580, -0.916183]).unsqueeze(0) 

    # Target Position (World Space)
    target_pos = torch.tensor([0.118386, -0.429724, 0.780205]).unsqueeze(0)

    print("-" * 20)
    print(f"Original Target Pos: {target_pos}")

    # 1. Calculate local offset
    local_offset = calculate_local_offset(knife_pos, knife_rot, target_pos)
    print(f"Calculated Local Offset: {local_offset}")

    # 2. Reconstruct world position from local offset
    reconstructed_pos = get_world_pos(knife_pos, knife_rot, local_offset)
    print(f"Reconstructed World Pos: {reconstructed_pos}")

    # 3. Verify error
    error = (reconstructed_pos - target_pos).abs().max().item()
    print(f"Max Error: {error}")

    if error < 1e-6:
        print("Test PASSED: Round-trip transformation successful.")
    else:
        print("Test FAILED: Error too large.")
    print("-" * 20)