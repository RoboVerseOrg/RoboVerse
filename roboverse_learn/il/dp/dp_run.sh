task_name=close_box
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
eval_max_step=500
expert_data_num=100
sim_set=mujoco
eval_ckpt_name=100          # Checkpoint epoch to evaluate

## Seperate training and evaluation
train_enable=True
eval_enable=True

## Choose training or inference algorithm
algo_choose=0  # 0: DDPM, 1: DDIM, 2: FM UNet 3: FM DiT  4: Score-based 5: VITA

algo_model=""
eval_path="./info/outputs/DP/${task_name}/checkpoints/${eval_ckpt_name}.ckpt"
case $algo_choose in
    0)
        # DDPM settings
        export algo_model="DDPM_model"
        ;;
    1)
        # DDIM settings
        export algo_model="DDIM_model"
        ;;
    2)
        # FM settings
        export algo_model="fm_model"
        ;;
    3)
        # FM DiT Settings
        export algo_model="fm_dit_model"
        ;;
    4)
        # Score-based settings
        export algo_model="score_model"
        ;;
    5)
        # VITA Settings
        export algo_model="vita_model"
        ;;
    *)
        echo "Invalid algorithm choice: $algo_choose"
        echo "Available options: 0 (DDPM), 1 (DDIM), 2 (FM UNet), 3 (FM DiT), 4 (Score-based), 5 (VITA)"
        exit 1
        ;;
esac

echo "Selected model: $algo_model"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python ./roboverse_learn/il/dp/main.py --config-name=${config_name}.yaml \
task_name="${task_name}" \
dataset_config.zarr_path="./data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path} \

# eval_config.eval_args.random.level=${level} \
