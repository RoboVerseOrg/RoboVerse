# ACT: Action Chunking with Transformers

### Installation
```bash
cd act/detr && pip install -e .
```

Move back to RoboVerse/ and run the following training script.
Similar to diffusion policy, it first stores the expert data in zarr format, and then trains a policy. You can configure joint or end effector control.
```bash
bash roboverse_learn/algorithms/act/train_act.sh <task_name> <robot> <expert_data_num> <level> <seed> <gpu_id> <DEBUG> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

### Example Usages
```bash
bash roboverse_learn/algorithms/act/train_act.sh CloseBox franka 100 0 42 0 3000 joint_pos joint_pos 0
```
