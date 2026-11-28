# BeyondMimic Motion Tracking

## Overview

BeyondMimic motion tracking code adapted for RoboVerse. Package structure:

- `roboverse_pack/tasks/beyondmimic/isaaclab`: Isaac Lab-native training.
- `roboverse_pack/tasks/beyondmimic/metasim`: simulator-agnostic (implemented using MetaSim handler) training (beta) and evaluation.
- `roboverse_pack/tasks/beyondmimic/scripts`: for converting and playing reference motions.

## Environment Setup

Tested with python 3.10, RSL-RL v3.1.0, and both Isaac Lab v2.1.0 and v2.3.0. You can install Isaac Lab following their [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installation-using-isaac-sim-pip-package), and install RSL-RL by:

```bash
git clone [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
git checkout v3.1.0
pip install -e .
```

## Motion Preparation

### Download Description Files

```bash
cd RoboVerse
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C roboverse_data/ && \
rm unitree_description.tar.gz
```

Since RoboVerse uses USD format to instantiate robots, you may need to convert URDF files to USD. We provide a utility script `roboverse_pack/tasks/beyondmimic/scripts/convert_urdf.py` for conversion. You can run it using:

```bash
python roboverse_pack/tasks/beyondmimic/scripts/convert_urdf.py {source-urdf-path} {target-usd-path} --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
```

Since currently the paths are hard-coded inside the robot config (`roboverse_pack/robots/g1_tracking.py`), you may replace `{source-urdf-path}` with `roboverse_data/unitree_description/urdf/g1/main.urdf` and `{target-usd-path}` with `roboverse_data/unitree_description/usd/g1/g1.usd`.

### Motion Preprocessing & Registry Setup

We leverage the WandB registry to store and load reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of Unitree's dataset.
    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP)
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body acceleration) via forward kinematics:

```bash
python roboverse_pack/tasks/beyondmimic/scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python roboverse_pack/tasks/beyondmimic/scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py to use a temporary folder of your choice.

## Policy Training

> **Retired on 2026-11-27.** The tracking tasks named below (`motion-tracking`, `motion-tracking-isaaclab`,
> `motion-tracking-isaaclab-deploy`) are no longer registered: the release gate could not run them without the
> WandB motion registry, so they are listed in `roboverse_pack/tasks/RETIRED_TASKS.md` with the commit to restore
> them from. The training code under `roboverse_pack/tasks/beyondmimic/` and `roboverse_learn/rl/rsl_rl/` is
> kept; the commands below work once the registrations are restored.


- Isaac Lab-Native

```bash
python roboverse_learn/rl/rsl_rl/ppo_tracking.py --task motion-tracking-isaaclab --sim isaacsim --num_envs 4096 --use-wandb --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} --headless --logger wandb
```

For training a tracker with delayed actuator and reduced observations (`base_lin_vel` and `motion_anchor_pos_b`), you can replace the command with `--task motion-tracking-isaaclab-deploy`.

- MetaSim Handler

Replace the above command with `--task motion-tracking`. Note that MetaSim version training is still in beta and is prone to issues. Feel free to pull a PR if you've improved the code or fixed any issues regarding this pipeline.

## Policy Evaluation

```bash
python roboverse_learn/rl/rsl_rl/eval_tracking.py --task motion-tracking --sim isaacsim --num-envs 2 --robot g1_tracking --wandb-path {wandb-run-path}
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along with a unique 8-character identifier. Note that run_name is different from run_path.

The evaluation pipeline is written using MetaSim handler and thus simulator-agnostic. However, currently simulators other than Isaac Sim may have missing functionality required by this pipeline, therefore may not behave as expected.