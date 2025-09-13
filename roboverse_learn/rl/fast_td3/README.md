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
