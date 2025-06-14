# SkillBlender RL
We provide implementent [SkillBlender](https://github.com/Humanoid-SkillBlender/SkillBlender) into our framework.

RL algorithm: `PPO` by [rsl_rl](https://github.com/leggedrobotics/rsl_rl) `v1.0.2`

RL learning framework: `hierarchical RL`

Simulator: `IsaacGym`

## Installation
```bash
pip install -e roboverse_learn/rl/rsl_rl
```

## Training

- IssacGym:
    ```bash
    python3 roboverse_learn/skillblender_rl/train_skillblender.py --task "skillblender:Walking" --sim "isaacgym" --num_envs 1024 --robot "h1_wrist" --use_wandb
   ```
    after training around a few minuts for task `skillblender:Walking` and `skillblender:Stepping`, you can see like this.
**To speed up training, click the IsaacGym viewer and press V to stop rendering.**
## Task list
> 4 Goal-Conditional Skills
- [x]  Walking
- [x]  Squatting
- [x]  Stepping
- [ ]  Reaching
> 8 Loco-Manipulation Tasks
- [ ]  FarReach
- [ ]  ButtonPress
- [ ]  CabinetClose
- [ ]  FootballShoot
- [ ]  BoxPush
- [ ]  PackageLift
- [ ]  BoxTransfer
- [ ]  PackageCarry

## Robots supports
- [x]  h1
- [ ]  g1

## Todos
- [ ] ground type selection
- [ ] pushing robot
- [ ] sim2sim

## How to add new Task
1. Create a new `wrapper.py` in , add reward function
    define your reward functions in reward_fun_cfg.py, check whether the current states is enough for reward computation. If not, parse your state as follow:
    ```
    def _parse_NEW_STATES(self, envstate):
        """NEWSTATES PARSEING..."""

    def _parse_state_for_reward(self, envstate):
        super()._parse_state_for_reward(self, envstate):
        _parse_NEW_STATES(self, envstate)
    ```
2. Implemented `_compute_observation()`
    - fill `obs` and `privelidged_obs`.
    - modified `_post_physics_step` to reset variables you defined with `reset_env_idx`


3. Add Cfg for your task `metasim/cfg/tasks/skillblender`


## References and Acknowledgements
We implement SkillBlender based on and inspired by the following projects:
- [SkillBlender](https://github.com/Humanoid-SkillBlender/SkillBlender)
- [Legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse/tree/master)
