## run nvidia-smi to check available GPUs
export CUDA_VISIBLE_DEVICES=0

## Parameters
task_name_set=close_box
random_level=2          # 0: No randomization 1: Randomize visual material 2: Randomize camera pose 3: Randomize object reflection and lighting
num_envs=1              # Number of parallel environments
demo_start_idx=0        # Index of the first demo to collect
max_demo_idx=1000       # Maximum index of demos to collect
sim_set=isaacsim
# sim_set=mujoco

obs_space=joint_pos
act_space=joint_pos
delta_ee=0              # Delta control
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

## Collecting demonstration data run_all,
# python ~/RoboVerse/scripts/advanced/collect_demo.py --sim=${sim_set} --task=${task_name_set} --num_envs=${num_envs} --run_all --headless --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx} \
# python ~/RoboVerse/scripts/advanced/collect_demo.py --sim=${sim_set} --task=${task_name_set} --num_envs=${num_envs} --run_all --headless --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx}  --enable-randomization\


## Convert demonstration data
python ~/RoboVerse/roboverse_learn/il/data2zarr_dp.py \
--task_name ${task_name_set}FrankaL${random_level}_${extra} \
--expert_data_num 100 \
--metadata_dir ~/RoboVerse/roboverse_demo/demo_${sim_set}/${task_name_set}-/robot-franka \
--action_space ${act_space} \
--observation_space ${obs_space}

# expert_data_num    Number of expert demonstrations to process
# metadata_dir       Path to the directory containing demonstration metadata saved by collect_demo
# action_space       Type of action space to use
# observation_space  Type of observation space to use
