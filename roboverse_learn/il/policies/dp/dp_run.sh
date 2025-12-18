
## Seperate training and evaluation
train_enable=True  # True for training, False for evaluation
eval_enable=True

task_name_set=close_box
config_name=default_runner
num_epochs=100
port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300
expert_data_num=100
sim_set=isaacsim
eval_ckpt_name=100           # Evaluate the last checkpoint (epoch 3)

## Domain Randomization Configuration
level=3              # 0=None, 1=Scene+Material, 2=+Light, 3=+Camera
scene_mode=3         # 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
dr_seed=42          # Random seed for reproducible DR (null for random)


## Choose training or inference algorithm
# Supported models:
#   "ddpm_unet", "ddpm_dit", "ddim_unet", "vita", "fm_dit", "fm_unet", "score"
policy_name="ddpm_dit"
if [[ "${policy_name}" != *_model ]]; then
  policy_name="${policy_name}_model"
fi
export policy_name
eval_path="./il_outputs/${policy_name}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Selected model: $policy_name"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# Note: level variable is now used for DR, not in zarr filename
# The zarr filename should use the data collection level (e.g., L0)
data_level=0  # Level used when collecting data
zarr_path="./data_policy/${task_name_set}FrankaL${data_level}_${extra}_${expert_data_num}.zarr"

python ./roboverse_learn/il/policies/dp/main.py --config-name=${config_name}.yaml \
task_name=${task_name_set} \
dataset_config.zarr_path="${zarr_path}" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name_set} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
+eval_config.eval_args.max_demo=${expert_data_num} \
+eval_config.eval_args.level=${level} \
+eval_config.eval_args.scene_mode=${scene_mode} \
+eval_config.eval_args.randomization_seed=${dr_seed} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path}
