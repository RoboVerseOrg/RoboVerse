task_name=close_box
level=2
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
sim_set=isaacsim

## Seperate training and evaluation
train_enable=False
eval_enable=True

## Choose training or inference algorithm
algo_choose=0  # 0: DDPM, 1: DDIM, 2: FM  3: Score-based

algo_model=""
eval_path=""
case $algo_choose in
    0)
        # DDPM settings
        export algo_model="DDPM_model"
        eval_path="/home/jjindou/RoboVerse/info/outputs/DP/2025.09.03/13.40.19_close_box_obs:joint_pos_act:joint_pos/checkpoints/100.ckpt"
        ;;
    1)
        # DDIM settings
        export algo_model="DDIM_model"
        eval_path="/home/jjindou/RoboVerse/info/outputs/DP/2025.09.03/13.40.19_close_box_obs:joint_pos_act:joint_pos/checkpoints/100.ckpt"
        ;;
    2)
        # FM settings
        export algo_model="fm_model"
        eval_path="/home/jjindou/RoboVerse/info/outputs/DP/2025.09.03/02.39.59_close_box_obs:joint_pos_act:joint_pos/checkpoints/100.ckpt"
        ;;
    3)
        # Score-based settings
        export algo_model="Score_model"
        eval_path="/home/jjindou/RoboVerse/info/outputs/DP/2025.09.03/02.39.59_close_box_obs:joint_pos_act:joint_pos/checkpoints/100.ckpt"
        ;;
    *)
        echo "Invalid algorithm choice: $algo_choose"
        echo "Available options: 0 (DDPM), 1 (DDIM), 2 (FM), 3 (Score)"
        exit 1
        ;;
esac

echo "Selected model: $algo_model"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python ~/RoboVerse/roboverse_learn/il/dp/main.py --config-name=${config_name}.yaml \
task_name="${task_name}_${extra}" \
dataset_config.zarr_path="data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path} \

# eval_config.eval_args.random.level=${level} \

## Seperate training and evaluation
# 1. open runner/dp_runner.py
# 2. only training: set `def run(self, train=True, eval=True, ckpt_path=None):` to
#                       `def run(self, train=True, eval=False, ckpt_path="None"):`
# 4. only evaluation: set `def run(self, train=True, eval=True, ckpt_path=None):` to
#                         `def run(self, train=False, eval=True, ckpt_path="/path/to/your/checkpoint.ckpt"):`
