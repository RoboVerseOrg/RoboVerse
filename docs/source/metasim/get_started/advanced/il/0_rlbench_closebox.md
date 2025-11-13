# 1. CloseBox

Imitation Learning is a powerful tool for training agents to perform tasks by leveraging expert demonstrations, especially when reward signals are sparse or hard to specify.

In this example, we will collect demonstration trajectories of a robotic arm performing various tasks in simulator, then train an agent using Behavioral Cloning to mimic the expert’s motions and generalize to unseen targets.

## Collecting Demonstration

To train the policy, we first need to collect demonstration for it to learn from. In this documentation, we use `CloseBox` as an example to collect demonstration trajectories.


### Task: Close Box
```bash
python collect_demo.py \
  --task CloseBox \
  --robot franka \
  --sim mujoco \
  --num_envs 1 \
  --headless True \
  --random.level 0 \
  --run_unfinished
```

Please check the [Collecting Demonstration](https://roboverse.wiki/metasim/user_guide/collect_demo_tutorial) for more details about how to collect demonstration and what are the options for the command.

### Training the policy and Converting the dataset format
After collected to demo data, we can transform into zarr folders so we can read them more easily to train the policy. In this demo, we use diffusion policy as an example. You can also use other algorithms under `roboverse_learn/algorithms`.
```bash
task_name=CloseBox
level=2
expert_data_num=100
gpu_id=0
num_epochs=200

bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" 100 "${gpus}" "${num_epochs}" joint_pos joint_pos 0 1 1
```


### Evaluating the policy
After training the policy, we can evaluate it in the environment. The following command will run the trained policy in the environment and save the video output to `tmp/${task_name}/${policy_name}/${robot_name}/`.
```bash
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --num_envs <up to ~50 envs works on RTX> --checkpoint_path <checkpoint_path>
```


### Example demos

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 48%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/il/0_closebox.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;" >CloseBox</p>
        </div>
    </div>
</div>
