# ACT

ACT (Action Chunking with Transformers) implements a transformer-based VAE policy, which generates chunks of ~100 actions at each step. These are averaged using temporal ensembling to generate a single action. This algorithm was introduced by the [Aloha](https://arxiv.org/abs/2304.13705) paper, and uses the same implementation.

## Installation

```bash
cd roboverse_learn/algorithms/act/detr
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
--observation_space <observation_space> \
--delta_ee <delta_ee>
```
| Argument | Description | Example |
|----------|-------------|---------|
| `task_name` | Name of the task | `CloseBoxFrankaL0_obs:joint_pos_act:joint_pos` |
| `expert_data_num` | Number of expert demonstrations to process | `100` |
| `metadata_dir` | Path to the directory containing demonstration metadata saved by collect_demo | `roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka` |
| `action_space` | Type of action space to use (options: 'joint_pos' or 'ee') | `joint_pos` |
| `observation_space` | Type of observation space to use (options: 'joint_pos' or 'ee') | `joint_pos` |
| `delta_ee` | (optional) Delta control (0: absolute, 1: delta; default 0) | `0` |

### **Training**:
_roboverse_learn/algorithms/act/train.py_ uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the ACT model.

Command:
```shell
python -m roboverse_learn.algorithms.act.train \
--task_name <task_name> \
--num_episodes <num_episodes> \
--dataset_dir <dataset_dir> \
--policy_class <policy_class> \
--kl_weight <kl_weight> \
--chunk_size <chunk_size> \
--hidden_dim <hidden_dim> \
--batch_size <batch_size> \
--dim_feedforward <dim_feedforward> \
--num_epochs <num_epochs> \
--lr <lr> \
--state_dim <state_dim> \
--seed <seed>
```
| Argument | Description | Example |
|----------|-------------|---------|
| `task_name` | Name of the task | `CloseBoxFrankaL0_obs:joint_pos_act:joint_pos` |
| `num_episodes` | Number of episodes in the dataset | `100` |
| `dataset_dir` | Path to the zarr dataset created in Data Preparation step | `data_policy/CloseBoxFrankaL0_obs:joint_pos_act:joint_pos_100.zarr` |
| `policy_class` | Policy class to use | `ACT` |
| `kl_weight` | Weight for KL divergence loss | `10` |
| `chunk_size` | Number of actions per chunk | `100` |
| `hidden_dim` | Hidden dimension size for the transformer | `512` |
| `batch_size` | Batch size for training | `8` |
| `dim_feedforward` | Feedforward dimension for transformer | `3200` |
| `num_epochs` | Number of training epochs | `2000` |
| `lr` | Learning rate | `1e-5` |
| `state_dim` | State dimension (action space dimension) | `9` |
| `seed` | Random seed for reproducibility | `42` |

## Option 2: Run with Single Command: train_act.sh

We further wrap the data preparation and training into a single command: `train_act.sh`. This ensures consistency between the parameters of the data preparation and training, especially the action space, observation space, and data directory.

```shell
bash roboverse_learn/algorithms/act/train_act.sh <metadata_dir> <task_name> <expert_data_num> <gpu_id> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

| Argument          | Description                                                 | Example |
|-------------------|-------------------------------------------------------------|---------|
| `metadata_dir`    | Path to the directory containing demonstration metadata saved by collect_demo | `roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka` |
| `task_name`       | Name of the task                                            | `CloseBoxFrankaL0` |
| `expert_data_num` | Number of expert demonstrations to use                      | `100` |
| `gpu_id`          | ID of the GPU to use                                        | `0` |
| `num_epochs`      | Number of training epochs                                   | `2000` |
| `obs_space`       | Observation space (`joint_pos` or `ee`)                     | `joint_pos` |
| `act_space`       | Action space (`joint_pos` or `ee`)                          | `joint_pos` |
| `delta_ee`        | Optional: Delta control (`0` absolute, `1` delta; default 0)| `0` |

**Example:**
```shell
bash roboverse_learn/algorithms/act/train_act.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 2000 joint_pos joint_pos
```

**Important Parameter Overrides:**
- Key hyperparameters including `kl_weight` (set to 10), `chunk_size` (set to 100), `hidden_dim` (set to 512), `batch_size` (set to 8), `dim_feedforward` (set to 3200), and `lr` (set to 1e-5) are set directly in `train_act.sh`.
- `state_dim` is set to 9 by default, which works for both Franka joint space and end effector space.
- Notably, `chunk_size` is the most important parameter, which is defaulted to 100 actions per step.

### Switching between Joint Position and End Effector Control

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.
- Note the original ACT paper uses an action joint space of 14, but we modify the code to allow a parameterized action dimensionality `state_dim` to be passed into the training python script, which we default to 9 for Franka joint space or end effector space.

## Evaluation

To deploy and evaluate the trained policy:

```bash
python roboverse_learn/eval.py --task CloseBox --algo ACT --num_envs <up to ~50 envs works on RTX> --checkpoint_path <save_directory>
```

Ensure that `<save_directory>` points to the directory containing your trained model checkpoint, which should get saved to `info/outputs/ACT/...`
