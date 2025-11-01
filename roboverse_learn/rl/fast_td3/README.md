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
