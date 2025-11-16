# ACT

ACT (Action Chunking with Transformers) implements a transformer-based VAE policy, which generates chunks of ~100 actions at each step. These are averaged using temporal ensembling to generate a single action. This algorithm was introduced by the [Aloha](https://arxiv.org/abs/2304.13705) paper, and uses the same implementation.

**Key features:**

1. Action chunking strategy is used to avoid compound error.
2. Temporal ensemble helps prevent jerky robot motion.
3. Generative model, i.e., conditional variational autoencoder, is employed to handle the stochastic property of huam data.

## Installation

```bash
cd roboverse_learn/il/utils/act/detr
pip install -e .
cd ../../../../../

pip install pandas wandb
```

## Workflow 

### Step 1: Collect and pre-processing data

```bash
./roboverse_learn/il/collect_demo.sh
```

**collect_demo.sh** collects demos, i.e., metadata, using `~/RoboVerse/scripts/advanced/collect_demo.py` and converts the metadata into Zarr format for efficient dataloading. This script can handle both joint position and end effector action and observation spaces.

**Outputs**: Metadata directory is stored in `metadata_dir`. Converted dataset is stored in `~/RoboVerse/data_policy`

#### Parameters:

| Argument            | Description                                                  | Example                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `task_name_set`     | Name of the task                                             | `close_box`                                                  |
| `sim_set`           | Name of the selected simulator                               | `isaacsim`                                                   |
| `num_demo_success`  | Number of successful demos to collect                        | `100`                                                        |
| `expert_data_num`   | Number of expert demos to evaluate                           | `100`                                                        |
| `metadata_dir`      | Path to the directory containing demonstration metadata saved by collect_demo | `~/RoboVerse/roboverse_demo/demo_isaacsim/close_box-/robot-franka` |
| `action_space`      | Type of action space to use (options: 'joint_pos' or 'ee')   | `joint_pos`                                                  |
| `observation_space` | Type of observation space to use (options: 'joint_pos' or 'ee') | `joint_pos`                                                  |
| `delta_ee`          | (optional) Delta control (0: absolute, 1: delta; default 0)  | `0`                                                          |
| `cust_name`         | User defined name                                            | `noDR`                                                       |


### Step 2: Training and evaluation

```bash
./roboverse_learn/il/act/act_run.sh
```

`act_run.sh` uses `roboverse_learn/il/utils/act/train.py` and the generated Zarr data, which gets stored in the `data_policy/` directory, to train the ACT model. Subsequently, `roboverse_learn/il/act/act_eval_runner.py` is utilized to evaluate the trained model.  

**Outputs**: Training result is stored in `~/RoboVerse/info/outputs/ACT`. Evaluation result is stored in `~/RoboVerse/tmp/act`

#### Parameters:

| Argument       | Description          | Example     |
| -------------- | -------------------- | ----------- |
| `task_name`    | Name of the task     | `close_box` |
| `gpu_id`       | ID of the GPU to use | `0`         |
| `train_enable` | Enable training      | `true`      |
| `eval_enable`  | Enable evaluation    | `true`      |

#### Training parameters:

| Argument          | Description                                               | Example                                                      |
| ----------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| `num_episodes`    | Number of episodes in the dataset                         | `100`                                                        |
| `dataset_dir`     | Path to the zarr dataset created in Data Preparation step | `data_policy/CloseBoxFrankaL0_obs:joint_pos_act:joint_pos_100.zarr` |
| `policy_class`    | Policy class to use                                       | `ACT`                                                        |
| `kl_weight`       | Weight for KL divergence loss                             | `10`                                                         |
| `chunk_size`      | Number of actions per chunk                               | `100`                                                        |
| `hidden_dim`      | Hidden dimension size for the transformer                 | `512`                                                        |
| `batch_size`      | Batch size for training                                   | `8`                                                          |
| `dim_feedforward` | Feedforward dimension for transformer                     | `3200`                                                       |
| `num_epochs`      | Number of training epochs                                 | `2000`                                                       |
| `lr`              | Learning rate                                             | `1e-5`                                                       |
| `state_dim`       | State dimension (action space dimension)                  | `9`                                                          |
| `seed`            | Random seed for reproducibility                           | `42`                                                         |


**Important Parameter Overrides:**

- Key hyperparameters including `kl_weight` (set to 10), `chunk_size` (set to 100), `hidden_dim` (set to 512), `batch_size` (set to 8), `dim_feedforward` (set to 3200), and `lr` (set to 1e-5) are set directly in `train_act.sh`.
- `state_dim` is set to 9 by default, which works for both Franka joint space and end effector space.
- Notably, `chunk_size` is the most important parameter, which is defaulted to 100 actions per step.

**Switching between Joint Position and End Effector Control**

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.
- Note the original ACT paper uses an action joint space of 14, but we modify the code to allow a parameterized action dimensionality `state_dim` to be passed into the training python script, which we default to 9 for Franka joint space or end effector space.

#### Evaluation parameters:

| Argument          | Description                                            | Example                                                      |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| `algo`     | Evaluation algorithm   | `act` |
| `num_envs` | Number of environments | `1`   |
| `num_eval` | Number of evaluated samples | `100`   |
| `checkpoint_path` | The directory containing your trained model checkpoint | `~/RoboVerse/info/outputs/ACT/2025.09.04/01.37.14_close_box_obs:joint_pos_act:joint_pos_100` |


**Outputs**: Training result is stored in `~/RoboVerse/info/outputs/ACT`. Evaluation result is stored in `~/RoboVerse/tmp/act`


## Initial test results
### Task: close_box
**Setup:** `temporal_agg=True, num_episodes=100, num_epochs=100`
| Chunking size | 成功率 |
| ------------- | ------ |
| 1             | 0.17   |
| 10            | 0.17   |
| 20            | 0.76   |
| 40            | 0.55   |
| 60            | 0.51   |
| 80            | 0.56   |
| 100           | 0.09   |
| 120           | 0.66   |
| 140           | 0.53   |
| 160           | 0.54   |

### Task: pick_butter
**Setup:** `temporal_agg=True, num_episodes=100, num_epochs=100`
| Chunking size | 成功率 |
| ------------- | ------ |
| 1             | 0      |
| 10            | 0.5    |
| 20            | 0      |
| 40            | 1      |
| 60            | 1      |
| 80            | 0      |
| 100           | 0      |
| 120           | 0      |
| 140           | 0.5    |
| 160           | 0      |