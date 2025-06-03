# SkillBench RL
We provide implementent [SkillBench](https://www.google.com/) into our framework.

RL algorithm: `PPO` by [rsl_rl](https://github.com/leggedrobotics/rsl_rl) v1.0.2

RL learning framework: `hierarchical RL`

Simulator: `IsaacGym`

## Installation
```bash
pip install -e roboverse_learn/skillblender_rl/rsl_rl
```

## Training

- IssacGym:
    ```bash
    python3 roboverse_learn/skillblender_rl/train.py --task "skillblender:Walking" --sim "isaacgym" --num_envs  124 --run_name "humanoid_skillblender" --use_wandb
   ```



## Task list
> 4 Goal-Conditional Skills
- [√] Walking
- [ ] Reaching
- [ ] Squatting
- [ ] Stepping
> 8 Loco-Manipulation Tasks
- [ ] FarReach
- [ ] ButtonPress
- [ ] CabinetClose
- [ ] FootballShoot
- [ ] BoxPush
- [ ] PackageLift
- [ ] BoxTransfer
- [ ] PackageCarry

## References and Acknowledgements
We implement SkillBench based on and inspired by the following projects:
- SkillBlender: Towards Versatile Humanoid Whole-Body Control via Skill Blending.
- [Legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse/tree/master)
