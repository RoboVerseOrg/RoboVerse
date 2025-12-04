# Pick Place Task Training Pipeline

Two-stage training pipeline for pick-and-place task using FastTD3.

## Overview

The task is split into two stages:
1. **Approach & Grasp**: Train robot to approach and grasp objects
2. **Track**: Train robot to track trajectory after successful grasp

## Training Pipeline

### Stage 1: Train Approach & Grasp

Train the first stage to learn approach and grasp:

```bash
cd roboverse_learn/rl/fast_td3
python train.py --config pick_place.yaml
```

This will generate checkpoints in the output directory. Note the checkpoint path for the next step.

### Stage 2: Evaluate Lift and Collect States

Evaluate the trained model and collect stable grasp states and first-half trajectories:

```bash
python evaluate_lift.py \
    --checkpoint models/pick_place.approach_grasp_simple_1210000.pt \
    --target_count 100 \
    --state_dir eval_states \
    --traj_dir eval_trajs
```

This generates:
- **States file**: `eval_states/pick_place.approach_grasp_simple_franka_lift_states_*.pkl` (stable grasp states)
- **Trajectories**: First-half trajectories in `eval_trajs/`

### Stage 3: Train Track Task

Load the collected states as initial states for track training:

```bash
python train.py --config track.yaml
```

Make sure `track.yaml` has the correct `state_file_path` pointing to the states file from Stage 2:

```yaml
state_file_path: "eval_states/pick_place.approach_grasp_simple_franka_lift_states_101states_20251122_180651.pkl"
```

### Stage 4: Evaluate Track

Evaluate the track task to get second-half trajectories:

```bash
python evaluate.py --checkpoint models/pick_place.track_*.pt
```

### Stage 5: Merge Trajectories

Combine the first-half trajectories (from Stage 2) and second-half trajectories (from Stage 4) to get complete task trajectories.


## Notes

- The track task starts from saved grasp states, training only the trajectory tracking phase
- States are collected when the object is successfully lifted and maintained stable for multiple frames
- Two trajectory segments are merged to form the complete pick-and-place task

