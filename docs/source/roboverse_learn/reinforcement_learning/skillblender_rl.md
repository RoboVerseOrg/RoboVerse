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
    python3 roboverse_learn/skillblender_rl/train.py --task "skillblender:Walking" --sim "isaacgym" --num_envs 1024 --robot "h1_wrist" --use_wandb
   ```
    after training around a few minuts for task `skillblender:Walking` and `skillblender:Stepping`, you can see like this. Note that we should always use `h1_wrist` instead of navie `h1` keep ths wrist links exist.
**To speed up training, click the IsaacGym viewer and press V to stop rendering.**
## Play
After training a few minutes, you can run the following play script
```
python3 roboverse_learn/skillblender_rl/play.py --task skillblender:Reaching --sim isaacgym --robot h1_wrist --load_run 2025_0628_232507  --checkpoint 15000
```
you can see video like this

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 48%; text-align: center;">
            <img src="https://roboverse.wiki/_static/standard_output/rl/2_skillblender_reaching.gif" style="width: 100%;" />
            <p style="margin-top: 5px;">Skillblender::Reaching</p>
        </div>
        <div style="width: 48%; text-align: center;">
            <img src="https://roboverse.wiki/_static/standard_output/rl/2_skillblender_walking.gif" style="width: 100%;" />
            <p style="margin-top: 5px;">Skillblender::Walking</p>
        </div>
    </div>
</div>

## Checkpoints
We also provide [checkpoints](https://huggingface.co/RoboVerseOrg/ckeckpionts/blob/main/skillblender_reaching_ckpt.pt
)  trained with roboverse humanoid infra. To use it with `roboverse_learn/skillblender_rl/play.py`,
rename the file to `model_xxx.pt` and move it into the appropriate directory, which should have the following layout:

```
outputs/
└── skillblender/
    └── h1_wrist_reaching/        # Task name
        └── 2025_0628_232507/     # Timestamped experiment folder
            ├── reaching_cfg.py   # Config snapshot (copied from metasim/cfg/tasks/skillblender)
            ├── model_0.pt        # Checkpoint at iteration 0
            ├── model_500.pt      # Checkpoint at iteration 500
            └── ...
```
when play

## Task list
> 4 Goal-Conditional Skills
- [x]  Walking
- [x]  Squatting
- [x]  Stepping
- [x]  Reaching
> 8 Loco-Manipulation Tasks
- [x]  FarReach
- [x]  ButtonPress
- [x]  CabinetClose
- [x]  FootballShoot
- [x]  BoxPush
- [x]  PackageLift
- [x]  BoxTransfer
- [x]  PackageCarry



## Robots supports
- [x]  h1
- [x]  g1
- [ ]  h1_2

## Todos
- [x] domain randomization
- [x] pushing robot
- [ ] sim2sim

## How to add new Task
1. **Create your wrapper module**
    - Add a new file `abc_wrapper.py` under `roboverse_learn/skillblender_rl/env_wrappers`
    - Add a config file `abc_cfg.py` under `metasim/cfg/tasks/skillblender`
    - define your reward functions in reward_fun_cfg.py, check whether the current states or variables are enough for reward computation.

2. If states not enough, add global variable by overriding `_init_buffer()`
    ```
    def _init_buffers(self):
        super()._init_buffers()
        """DEFINED YOUR VARIABLE or BUFFER HERE"""
        self.xxx = xxx
    ```
3. parse your state for reward computation if necessary:
    ```
    def _parse_NEW_STATES(self, envstate):
        """NEWSTATES PARSEING..."""
        envstate[robot_name].extra{'xxx'} = self.xxx

    def _parse_state_for_reward(self, envstate):
        super()._parse_state_for_reward(self, envstate):
        _parse_NEW_STATES(self, envstate)
    ```
3. Implemented `_compute_observation()`
    - fill `obs` and `privelidged_obs`.
    - modified `_post_physics_step` to reset variables you defined with `reset_env_idx`


3. Add Cfg for your task `metasim/cfg/tasks/skillblender`


## References and Acknowledgements
We implement SkillBlender based on and inspired by the following projects:
- [SkillBlender](https://github.com/Humanoid-SkillBlender/SkillBlender)
- [Legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse/tree/master)
