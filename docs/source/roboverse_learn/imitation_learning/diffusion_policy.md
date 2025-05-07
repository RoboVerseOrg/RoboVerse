# Diffusion Policy

## Installation

```bash
cd roboverse_learn/algorithms/diffusion_policy
pip install -e .
cd ../../../

pip install pandas wandb
```


## Option 1: Two Step, Pre-processing and Training

### **Data Preparation**:
_data2zarr_dp.py_ converts the metadata stored by the collect_demo script into Zarr format for efficient dataloading. This script can handle both joint position and end effector action and observation spaces.

Command:
```shell
python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name <task_name> \
--expert_data_num <expert_data_num> \
--metadata_dir <metadata_dir> \
--action_space <action_space> \
--observation_space <observation_space>
```
| Argument | Description | Example |
|----------|-------------|---------|
| `task_name` | Name of the task | `CloseBox_Franka_Level0_obs:joint_pos_action:joint_pos` |
| `expert_data_num` | Number of expert demonstrations to process | `100` |
| `metadata_dir` | Path to the directory containing demonstration metadata saved by collect_demo | `roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka` |
| `action_space` | Type of action space to use (options: 'joint_pos' or 'ee') | `joint_pos` |
| `observation_space` | Type of observation space to use (options: 'joint_pos' or 'ee') | `joint_pos` |
| `delta_ee` | (optional) Delta control (0: absolute, 1: delta; default 0) | `0` |

### **Training**:
_diffusion_policy/train.py_ uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the diffusion policy model. Note the policy.runner arguments should match the arguments used in _data2zarr_dp.py_ and are used in downstream evaluations.

Command:
```shell
python roboverse_learn/algorithms/diffusion_policy/train.py \
--config-name=robot_dp.yaml \
task.name=<task_name> \
task.dataset.zarr_path=<zarr_path> \
training.seed=<seed> \
horizon=<horizon> \
n_obs_steps=<n_obs_steps> \
n_action_steps=<n_action_steps> \
training.num_epochs=<num_epochs> \
policy_runner.obs.obs_type=<obs_type> \
policy_runner.action.action_type=<action_type> \
policy_runner.action.delta=<delta> \
training.device=<device>
```
| Argument | Description | Example |
|----------|-------------|---------|
| `task_name` | Name of the task | `CloseBox_Franka_Level0_obs:joint_pos_action:joint_pos` |
| `zarr_path` | Path to the zarr dataset created in Step 1. This will be {task_name}_{expert_data_num}.zarr | `data_policy/CloseBox_Franka_Level0_obs:joint_pos_action:joint_pos_100.zarr` |
| `seed` | Random seed for reproducibility | `42` |
| `horizon` | Time horizon for the policy | `8` |
| `n_obs_steps` | Number of observation steps | `3` |
| `n_action_steps` | Number of action steps | `4` |
| `num_epochs` | Number of training epochs | `200` |
| `obs_type` | Observation type (joint_pos or ee) | `joint_pos` |
| `action_type` | Action type (joint_pos or ee) | `joint_pos` |
| `delta` | Delta control mode (0 for absolute, 1 for delta) | `0` |
| `device` | GPU device to use | `"cuda:7"` |

## Option 2: Run with Single Command: train_dp.sh

We further wrap the data preparation and training into a single command: `train_dp.sh`. This ensures consistency between the parameters of the data preparation and training, especially the action space, observation space, data directory.

```shell
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh <metadata_dir> <task_name> <expert_data_num> <gpu_id> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

| Argument          | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `metadata_dir`    | Path to the directory containing demonstration metadata saved by collect_demo |
| `task_name`       | Name of the task                                            |
| `expert_data_num` | Number of expert demonstrations to use         |
| `gpu_id`          | ID of the GPU to use                                        |
| `num_epochs`      | Number of training epochs                                   |
| `obs_space`       | Observation space (`joint_pos` or `ee`)                     |
| `act_space`       | Action space (`joint_pos` or `ee`)                          |
| `delta_ee`        | Optional: Delta control (`0` absolute, `1` delta; default 0)|

**Example:**
```shell
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 200 joint_pos joint_pos
```

**Important Parameter Overrides:**
- `horizon`, `n_obs_steps`, and `n_action_steps` are set directly in `train.sh` and override the YAML configurations.
- All other parameters (e.g., batch size, number of epochs) can be manually adjusted in the YAML file: `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/robot_dp.yaml`
- If you alter observation and action spaces, verify the corresponding shapes in: `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/task/default_task.yaml` Both end effector control and Franka joint space, have dimension 9 but keep this in mind if using a different robot.

### Switching between Joint Position and End Effector Control

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.

Adjust relevant configuration parameters in:
- `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/robot_dp.yaml`


## Evaluation

To deploy and evaluate the trained policy:

```bash
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --num_envs <up to ~50 envs works on RTX> --checkpoint_path <checkpoint_path>
```

Ensure that `<checkpoint_path>` points to the file of the trained model checkpoint, ie `info/outputs/DP/...`
