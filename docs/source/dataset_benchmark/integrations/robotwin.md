# RoboTwin → MetaSim / Sapien integration

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is a 50-task
dual-arm tabletop benchmark built on SAPIEN 3.0.0b1 + mplib + curobo.
Tasks live under `envs/<task>.py` and each declares its own scene
setup, success criterion, and scripted-policy data collector via raw
SAPIEN API.

## Status

- **What works**: RoboTwin's ALOHA-AgileX embodiment
  (`arx5_description_isaac.urdf`, 38 DoF: dual 6-DoF arms with 2-finger
  mimic grippers + mobile base + sensor mast) loads and steps in
  MetaSim/Sapien3 after one small handler fix
  (`fix/sapien3-passive-joints`).
- **What's deferred**: per-task scene/reward ports — RoboTwin's task
  code requires SAPIEN 3.0.0b1 (we run 3.0.3), mplib 0.2.1 (we run
  0.1.1), and a separate curobo install. Per-task ports are 2-4 hrs
  each → 4-8 person-weeks for the full 50.

## MetaSim fix that enables this

The `Sapien3Handler` used to crash with `KeyError` when an active
URDF joint wasn't enumerated in `RobotCfg.actuators`. That's the rule
for most clean academic robots but it's wrong for any
embodiment that bundles wheels, suspension, or a sensor mast — those
DoFs exist in the URDF but no one wants them in the actuator dict.

The fix (`fix/sapien3-passive-joints`) switches the lookup to
`actuators.get(name)` and skips undriven joints. `default_joint_positions`
gets the same treatment, defaulting to 0.0 for unenumerated joints.
Two-line change in `_build_sapien`, plus a regression test at
`metasim/test/test_sapien3_passive_joints.py`.

## Asset layout

| Bundle                  | Size       | Needed?                                          |
|-------------------------|------------|--------------------------------------------------|
| `embodiments.zip`       | 220 MB     | **Yes** — robot URDFs + meshes for all 5 robots |
| `objects.zip`           | 3.74 GB    | Yes for task scene actors (YCB-style)            |
| `background_texture.zip`| 11 GB      | Domain-randomization training only               |
| Full dataset            | 1.47 TB    | Demo trajectories + RL checkpoints — not needed for sim parity |

Locator (`roboverse_pack/robots/aloha_agilex_cfg.py`) searches
`~/projects/robotwin/assets/` or `$ROBOTWIN_ASSETS`.

## Integration plan

Three options:

1. **Port each task** into `roboverse_pack/tasks/robotwin/`.
   - For each `envs/<name>.py`, extract the scene actors + poses + the
     `check_success()` body; re-express as a `BaseTaskEnv` subclass
     with a hand-built `ScenarioCfg`.
   - Reuse our existing curobo scripts (`scripts/curobo/`) for the
     scripted policy, or replay a downloaded demo.
   - **Preferred**, but expensive at full scope.
2. **Process-boundary wrapper**: install RoboTwin's exact deps in a
   sibling `robotwin` conda env, drive a task remotely from MetaSim
   via a thin RPC, mirror the scene snapshot. Lets us compare physics
   step-for-step without porting task logic.
3. **Trajectory replay**: pre-record N demonstrations per task in the
   `robotwin` env, save as roboverse-format trajectory files, replay
   them in MetaSim via the existing `get_traj` path.

## How to reproduce the embodiment load

```bash
mkdir -p ~/projects && cd ~/projects
git clone --depth 1 https://github.com/RoboTwin-Platform/RoboTwin.git robotwin
cd robotwin/assets
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('TianxingChen/RoboTwin2.0', allow_patterns=['embodiments.zip'], \
                    local_dir='.', repo_type='dataset')"
unzip -q embodiments.zip

cd "$METASIM"
git checkout fix/sapien3-passive-joints

cd "$ROBOVERSE"  # repo root
PYTHONPATH="$ROBOVERSE:$METASIM" \
  python -m tools.robotwin_integration.aloha_demo
```

Artefact: `reports/robotwin_integration/
aloha_demo_summary.json` with the load summary (38 DoF / 38 active
joints / 60 steps OK).
