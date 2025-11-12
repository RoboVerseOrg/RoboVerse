# Diffusion Policy

## Installation

```bash
cd roboverse_learn/il/utils/diffusion_policy

pip install -e .

cd ../../../../

pip install pandas wandb
```

Register for a Weights & Biases (wandb) account to obtain an API key.

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
./roboverse_learn/il/dp/dp_run.sh
```

`dp_run.sh` uses `roboverse_learn/il/dp/main.py` and the generated Zarr data, which gets stored in the `data_policy/` directory, to train and evaluate the DP model.

**Outputs**: Training result is stored in `~/RoboVerse/info/outputs/DP`. Evaluation result is stored in `~/RoboVerse/tmp`

#### Parameters:

| Argument       | Description          | Example     |
| -------------- | -------------------- | ----------- |
| `task_name`    | Name of the task     | `close_box` |
| `sim_set`      | Name of the selected simulator | `isaacsim` |
| `gpu_id`       | ID of the GPU to use | `0`         |
| `train_enable` | Enable training      | `True`      |
| `eval_enable`  | Enable evaluation    | `True`      |
| `algo_choose`  | Choose training or inference algorithm. 0: DDPM, 1: DDIM, 2: Flow matching  3: Score-based   | `0` |
| `eval_path`    | Path of trained DP model. Need to be set if  `train_enable = False`. | `/path/to/your/checkpoint.ckpt`|
| `zarr_path` | Path to the zarr dataset.| `data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr` |
| `seed` | Random seed for reproducibility | `42` |
| `num_epochs` | Number of training epochs | `200` |
| `obs_type` | Observation type (joint_pos or ee) | `joint_pos` |
| `action_type` | Action type (joint_pos or ee) | `joint_pos` |
| `delta` | Delta control mode (0 for absolute, 1 for delta) | `0` |
| `device` | GPU device to use | `"cuda:7"` |


**Important Parameter Overrides:**
- `horizon`, `n_obs_steps`, and `n_action_steps` are set directly in `dp_runner.sh` and override the YAML configurations.

**Switching between Joint Position and End Effector Control**

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.

Adjust relevant configuration parameters in:
- `roboverse_learn/il/dp/config/.yaml`
