# `mjlab → MetaSim` policy-replay + native-training harness

Tools to (a) train mjlab RL policies, (b) prove they run on MetaSim via
action replay, and (c) train mjlab tasks **inside MetaSim** with our own
PPO. All built 2026-05-17.

## What lives here

| File | Purpose |
|---|---|
| `run_mjlab_policy.py` | Run an mjlab task with a checkpointed policy; record per-step `ctrl` + `qpos` + `qvel` + compiled scene; render left-side mp4 from `env.unwrapped.render()`. |
| `replay_in_metasim.py` | Load the compiled scene from above; replay the recorded `ctrl` sequence in pure MuJoCo (MetaSim path); render right-side mp4 + diagnostic `qpos` drift vs mjlab native. |
| `stitch_side_by_side.py` | `ffmpeg hstack` two mp4s with title overlays. |
| `render_one_task.sh` | One-task end-to-end driver (mjlab native rollout → MetaSim replay → stitch). |
| `train_all_missing.sh` | Sequentially train + render all mjlab tasks lacking checkpoints. Idempotent (skips already-rendered tasks). |
| `extract_curves.py` | Read RSL-RL tensorboard event files for each trained task, plot reward / loss / entropy / lr panels per task, write `training_summary.json`. |
| `train_metasim_cartpole.py` | **MetaSim-native** PPO trainer (CleanRL-style, 350-line). Drives MetaSim's MuJoCo handler directly through `roboverse_pack.tasks.mjlab.cartpole_train.MjlabCartpole{Balance,Swingup}Train`. No mjlab runtime used. |
| `plot_metasim_curve.py` | Plot MetaSim-native training curve from `training_log.jsonl` + overlay mjlab native reference. |

## Three experiment types

**1. Physics step parity (already documented elsewhere)** — give same MJCF
+ same `ctrl` sequence to mjlab's mujoco-warp and MetaSim's pure-MuJoCo
handler; verify qpos/qvel diverge ≤1e-9 over 200 control steps. Covered
by the 12-task parity table in the main report.

**2. Policy rollout via action replay** — `render_one_task.sh` driver:
- mjlab native loads the policy, runs the env, records the
  post-action-manager `ctrl` trajectory + dumps the compiled scene
  (`compiled_scene.mjb`).
- `replay_in_metasim.py` loads that same compiled scene in pure MuJoCo
  and replays the ctrl sequence step-by-step.
- ffmpeg stitches both into a side-by-side mp4.
- Cumulative `qpos` drift between the two sides typically 1–10 rad over
  400 steps (mujoco-warp vs pure-mujoco float-precision accumulation);
  visually still the same behavior.

**3. MetaSim-native training (the strongest claim)** — train a cartpole
PPO policy entirely inside our framework (`train_metasim_cartpole.py`).
Uses `roboverse_pack.tasks.mjlab.cartpole_train` which ports
obs/action/reward/termination/reset 1:1 from mjlab to MetaSim. The PPO
loop is a minimal CleanRL-style implementation with adaptive lr + KL
control + ELU + init_std=1.0 to match mjlab's RSL-RL setup.

## Quick start

```bash
source /home/ghr/miniconda3/etc/profile.d/conda.sh && conda activate roboverse

# (1) Action-replay one task. Uses mjlab public ckpt for Tracking-Flat-G1.
cd /home/ghr/projects/RoboVerse/RoboVerse/tools/mjlab_integration/policy_replay
STEPS=400 HEIGHT=360 WIDTH=640 bash render_one_task.sh \
  Mjlab-Tracking-Flat-Unitree-G1 \
  /tmp/mjlab_cache/demo_ckpt.pt \
  /tmp/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz

# Rough-terrain task needs higher contact buffer:
NCONMAX=1024 STEPS=400 bash render_one_task.sh \
  Mjlab-Velocity-Rough-Unitree-G1 \
  /path/to/g1_velocity_rough/model_1999.pt

# (2) Train all mjlab tasks lacking ckpts; renders side-by-side after each.
bash train_all_missing.sh

# (3) MetaSim-native train of cartpole_balance (no mjlab runtime).
PYTHONPATH=/home/ghr/projects/RoboVerse/RoboVerse MUJOCO_GL=egl \
  python -m tools.mjlab_integration.policy_replay.train_metasim_cartpole \
    --task mjlab.cartpole_balance_train \
    --num-envs 64 --total-iter 4000 --num-steps 32 \
    --activation elu --init-std 1.0 --obs-norm 0 --lr-schedule adaptive

# Plot the curve:
python /home/ghr/projects/RoboVerse/RoboVerse/tools/mjlab_integration/policy_replay/plot_metasim_curve.py
```

## Output layout

```
/home/ghr/projects/RoboVerse/reports/mjlab_integration/runs/<task>/
  ├── mjlab_native.mp4              # left side
  ├── metasim_replay.mp4            # right side
  ├── policy_side_by_side.mp4       # stitched
  ├── trajectory.npz                # ctrl + qpos + qvel from mjlab native
  ├── metasim_replay.npz            # same from MetaSim replay
  ├── compiled_scene.mjb            # mjlab's exact compiled scene (excluded from publish)
  ├── training_curve.png            # if mjlab native trained the task here
  └── metasim_native_training_curve.png  # if MetaSim-native trained too

/home/ghr/projects/RoboVerse/RoboVerse/training_logs/
  ├── rsl_rl/<task>/<timestamp>/    # mjlab RSL-RL checkpoints + tensorboard
  └── metasim_native/<task>/        # MetaSim-native CleanRL training_log.jsonl + agent_final.pt
```

## Common gotchas

- **Conda env**: `roboverse`. Most scripts source it automatically (`render_one_task.sh`).
- **MUJOCO_GL=egl** required for headless rendering. Scripts set it.
- **GPU**: mjlab uses mujoco-warp (heavy GPU). MetaSim-native trainer uses CPU mujoco + GPU policy (light). Run mjlab training serially, then MetaSim-native concurrently.
- **PYTHONPATH** must include `/home/ghr/projects/RoboVerse/RoboVerse` for the `tools.mjlab_integration.*` modules to resolve.
- **Wandb**: disabled via `--agent.logger tensorboard`. If you don't pass that, training crashes with "No API key configured".
- **Tracking tasks**: training needs `--env.commands.motion.motion-file /path/to/motion.npz`. `train_all_missing.sh` passes this automatically.
- **Rough terrain rendering**: needs `NCONMAX=1024` (mujoco-warp default of 35 overflows).
- **Pickle buffering**: when running long-form training, prefer `python -u` + `PYTHONUNBUFFERED=1` so `[iter ...]` lines stream live to the log.

## See also

- Main report: `/home/ghr/projects/RoboVerse/reports/roboverse_overview/doc/INT_MJLAB_PARITY.html`
  — full results table + side-by-side videos + training curves + the "matches original repo" honest discussion.
- Task ports: `roboverse_pack/tasks/mjlab/cartpole_train.py` (the trainable variant; the existing
  `cartpole.py` is physics-only thin wrapper used for parity-test only).
