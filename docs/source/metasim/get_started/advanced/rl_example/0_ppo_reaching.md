# 0. PPO Reaching

RL is a powerful tool for training agents to perform tasks in simulation, expecially when we have large scale parallel simulation environments.

In this example, we will train a PPO agent to reach as far away as possible and also reach a target position in a 3D environment.

## One Command to Train PPO, Inference and Save Video
We provide tutorials for training PPO, inference and saving video. In this example, we will use stable baseline 3 to train PPO.

### Task: Reach Far Away

```bash
python get_started/rl/0_ppo_reaching.py --sim <simulator> --task debug:reach_far_away --num_envs <num_envs> --headless
```

### Task: Reach Target

```bash
python get_started/rl/0_ppo_reaching.py --sim <simulator> --task debug:reach_origin --num_envs <num_envs> --headless
```

## Example Commands and Results

### Task: Reach Far Away
Isaac Gym:
```bash
python get_started/rl/0_ppo_reaching.py --sim isaacgym --task debug:reach_far_away --num_envs 128 --headless
```

Isaac Lab:
```bash
python get_started/rl/0_ppo_reaching.py --sim isaaclab --task debug:reach_far_away --num_envs 128 --headless
```
### Task: Reach Target
Isaac Gym:
```bash
python get_started/rl/0_ppo_reaching.py --sim isaacgym --task debug:reach_origin --num_envs 128 --headless
```

Isaac Lab:
```bash
python get_started/rl/0_ppo_reaching.py --sim isaaclab --task debug:reach_origin --num_envs 128 --headless
```

### You can get the video like this:
#### Reach Far Away:
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 48%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/rl/0_ppo_reaching_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 48%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/rl/0_ppo_reaching_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Lab</p>
        </div>
    </div>
</div>

#### Reach Origin:
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 48%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/rl/0_ppo_reaching_ReachOrigin_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 48%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/rl/0_ppo_reaching_ReachOrigin_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Lab</p>
        </div>
    </div>
</div>
