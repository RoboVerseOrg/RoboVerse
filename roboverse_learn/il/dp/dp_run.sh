
## Seperate training and evaluation
train_enable=True  # True for training, False for evaluation
eval_enable=True

task_name_set=stack_cube
level=0
config_name=dp_runner
num_epochs=100              # Number of training epochs
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
eval_ckpt_name=100


## Choose training or inference algorithm
# Supported models:
#   "ddpm_unet_model", "ddpm_dit_model", "ddim_unet_model", "fm_unet_model", "fm_dit_model", "score_model", "vita_model"
export algo_model="ddpm_dit_model"
eval_path="./info/outputs/DP/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Selected model: $algo_model"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python ./roboverse_learn/il/dp/main.py --config-name=${config_name}.yaml \
task_name=${task_name_set} \
dataset_config.zarr_path="./data_policy/${task_name_set}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
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
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path} \

# eval_config.eval_args.random.level=${level} \
