# ManiSkill ↔ MetaSim / Sapien integration

[ManiSkill3](https://github.com/haosulab/ManiSkill) is a SAPIEN-backed
manipulation benchmark with dozens of tabletop tasks. RoboVerse's
MetaSim already speaks SAPIEN3, so the integration goal is task-level
parity: a `roboverse_pack.tasks.maniskill.*` task should behave like
the ManiSkill `gym.make("...-v1")` env it mirrors.

Live report (with embedded videos and gap analysis):
<http://localhost:8000/#roboverse/maniskill_integration>.

## Status

- **Harness**: `tools/maniskill_integration/` mirrors the mjlab harness
  layout — `inventory.py` catalogs 4 tabletop tasks; `maniskill_rollout.py`
  drives `gym.make` and captures actions / reward / success flags /
  qpos / qvel / object pose / goal pose / TCP / rgb frames;
  `run_sweep.py` writes per-task `runs/<task>/maniskill.{npz,mp4}` and
  `summary.json`.
- **MetaSim side**: the existing `roboverse_pack.tasks.maniskill.
  pick_cube.PickCubeTask` instantiates + steps on Sapien3 (`nq=9`,
  `T=20`, `final_cube_z=0.0198 m` under zero-delta target). It is a
  RoboVerse-flavored task, not a 1:1 reimplementation of PickCube-v1.
- **Gap to numeric parity**: 10 items, all localised to one task
  module. See the [gap analysis section](maniskill-gap-analysis)
  below.

## Coverage

| Gym ID        | Notes                                       |
|---------------|---------------------------------------------|
| `PickCube-v1` | Lift a 4 cm cube into a 2.5 cm goal sphere |
| `PushCube-v1` | Push a cube to a target region              |
| `StackCube-v1`| Stack red cube on green cube                |
| `PullCube-v1` | Pull a cube to a target zone                |

All four roll out + render under the harness; zero-action reward sums
range 2.3–5.4 (no task is succeeded by a stationary robot, as expected).

(maniskill-gap-analysis)=
## What it takes to reach 1:1 parity

`roboverse_pack/tasks/maniskill/pick_cube.py` needs:

1. **Cube** — change `size=(0.04,0.04,0.04)` to `size=(0.02,0.02,0.02)`
   (PickCube uses *half-size* 0.02 → 4 cm box overall). Drop the
   explicit mass and let SAPIEN's default density govern.
2. **Goal sphere** — add a kinematic `PrimitiveSphereCfg("goal",
   radius=0.025, no collision)`. Reset position to
   `cube.pos + (0, 0, U(0, 0.3))`.
3. **Robot** — fork the Franka asset into `panda_v2` with finger
   friction 2.0 (ManiSkill's custom Panda).
4. **Initial qpos** — override `default_joint_positions` to ManiSkill's
   pose (joint2 = π/8, joint4 = −5π/8, joint6 = 3π/4). Add `N(0, 0.02²)`
   noise inside `_get_initial_states`.
5. **Controller** — implement `pd_joint_delta_pos`: 7-D arm delta
   (clipped to ±1, scaled by 0.1) + 1-D gripper mimic.
6. **Reset distribution** — sample cube XY in `U([-0.1, 0.1]²)`,
   z-only rotation; goal Z in `cube.z + U(0, 0.3)`. Seed from `env.reset()`.
7. **Dense reward** — port the `reaching + grasp + place·grasp +
   static·placed` formula. `is_grasped` needs SAPIEN contact queries;
   the handler lacks a public method but the underlying scene exposes
   `get_contacts()` — encapsulate inside the task module per the
   no-framework-edits rule.
8. **Success** — `(cube → goal ≤ 0.025) AND (max|qvel[:-2]| < 0.2)`;
   grasping NOT required at success.
9. **Episode length** — 50 steps, not 250.

Each item ≈ 30–90 min of focused edit; ≈ 1 day total for PickCube-v1.

## Known Sapien3 handler gaps

(Pulled forward from the [recon report](http://localhost:8000/#roboverse/maniskill_sapien_repro_2026_05_16).)

- No contact-force query on `BaseSimHandler` — ManiSkill's `is_grasped`
  needs pairwise contacts at both finger pads.
- Single-env only; `env_ids` silently ignored.
- No seeding hook; task layer must seed numpy + sapien scene in
  `reset()`.
- Scenario lights ignored (hard-coded ambient + 3 point lights).

## How to reproduce

```bash
conda activate roboverse
pip install mani_skill  # 3.0.1

cd "$ROBOVERSE"  # repo root
PYTHONPATH="$ROBOVERSE:$METASIM" \
  python -m tools.maniskill_integration.run_sweep --n-steps 50 --seed 0
```

Artefacts: `reports/maniskill_integration/`
(`maniskill_summary.json`, plus `runs/<task>/maniskill.{npz,mp4}` and
`summary.json`).
