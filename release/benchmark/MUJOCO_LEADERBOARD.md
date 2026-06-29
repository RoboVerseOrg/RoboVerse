# RoboVerse Task Benchmark + Leaderboard (MuJoCo)

**Setup:** 8× RTX 5090, headless EGL, MuJoCo 3.10, `/venv/roboverse` torch 2.10+cu128, our MetaSim+RoboVerse `main`.
**Scope:** 345 task classes sampled across **every** family (maniskill capped at 60; full coverage for the rest), run 24-way parallel.
**Protocol:** manipulation = **expert demo-replay** (load recorded trajectory, replay actions, success = task checker fires); locomotion/RL = zero-action rollout (mean reward; `term` = episode ended).

## TL;DR
- **57 / 339 task classes (16%) actually run on MuJoCo.** The rest are **backend-locked**: 137 ship only `usd`/`urdf` assets (Isaac/SAPIEN) with **no `mjcf`**, so MuJoCo can't build them; others miss assets in the HF dataset or hit code gaps.
- **9 / 51 demo-replays reproduce success on MuJoCo** — cross-sim demo parity is low (demos were recorded on other backends; MuJoCo dynamics differ). `libero` & `libero_90` load best (10/10, 31/65).
- **RL/locomotion**: `humanoid`/`reach` step and return rewards; `g1`/`h1` rollouts terminate early (fall) under a zero policy (expected — no trained controller).
- Caveat: MuJoCo physics is **CPU-stepped**; the 8 GPUs only drove parallel EGL rendering. Saturating GPUs for *physics* needs MJX/IsaacGym, which are **not installed** in this venv.

---

Tasks attempted: **339** across **20** families. Manipulation = expert **demo-replay** success; locomotion/RL = zero-action rollout (reward + early-term).

## Leaderboard by family

| Family | Tasks | Loaded on MuJoCo | Replay tasks | **Replay success** | RL tasks | Mean RL reward |
|---|---:|---:|---:|---:|---:|---:|
| libero_90 | 65 | 31/65 (47%) | 31 | 5/31 (16%) | 0 | — |
| maniskill | 60 | 4/60 (6%) | 4 | 2/4 (50%) | 0 | — |
| libero | 10 | 10/10 (100%) | 10 | 1/10 (10%) | 0 | — |
| rlbench | 79 | 6/79 (7%) | 6 | 1/6 (16%) | 0 | — |
| humanoid | 9 | 4/9 (44%) | 0 | — | 4 | 0.073 |
| reach | 2 | 2/2 (100%) | 0 | — | 2 | -0.153 |
| agibot_a2 | 1 | 0/1 (0%) | 0 | — | 0 | — |
| bimanual | 1 | 0/1 (0%) | 0 | — | 0 | — |
| calvin | 5 | 0/5 (0%) | 0 | — | 0 | — |
| box_task | 1 | 0/1 (0%) | 0 | — | 0 | — |
| embodiedgen | 2 | 0/2 (0%) | 0 | — | 0 | — |
| robosuite | 5 | 0/5 (0%) | 0 | — | 0 | — |
| mjlab | 23 | 0/23 (0%) | 0 | — | 0 | — |
| libero_native | 50 | 0/50 (0%) | 0 | — | 0 | — |
| beyondmimic | 1 | 0/1 (0%) | 0 | — | 0 | — |
| motion-tracking-isaaclab | 1 | 0/1 (0%) | 0 | — | 0 | — |
| motion-tracking-isaaclab-deploy | 1 | 0/1 (0%) | 0 | — | 0 | — |
| mujoco_playground | 3 | 0/3 (0%) | 0 | — | 0 | — |
| pick_place | 19 | 0/19 (0%) | 0 | — | 0 | — |
| xhand | 1 | 0/1 (0%) | 0 | — | 0 | — |
| **TOTAL** | **339** | **57/339 (16%)** | **51** | **9/51** | **6** | |

## Top failure reasons (why tasks didn't run on MuJoCo)

- **137×** ValueError: Object '…' has no mjcf asset path set for the '…' simulato
- **55×** AttributeError: type object '…' has no attribute '…'
- **17×** Exception: File roboverse_data/assets/libero/COMMON/stable_scanned_obj
- **12×** TypeError: ManagerBasedRVEnv.reset() got an unexpected keyword argumen
- **11×** Exception: File roboverse_data/assets/libero/COMMON/turbosquid_objects
- **11×** TypeError: object of type '…' has no len()
- **7×** Exception: File roboverse_data/assets/EmbodiedGenData/demo_assets/tabl
- **7×** AlreadyLocked: [Errno 11] Resource temporarily unavailable
- **4×** RuntimeError: indices should be either on cpu or on the same device as
- **3×** FileNotFoundError: State file not found: eval_states/pick_place.approa
- **2×** TypeError: TrackingRLEnv.__init__() missing 1 required positional argu
- **2×** KeyError: "Invalid name '…'. Valid names: ['…', '…', '…', '…', 'franka

## Tasks that succeeded (replay success or RL early-termination)

| Task | Mode | Steps/DemoLen | Time(s) |
|---|---|---|---:|
| g1.walk_g1_dof12 | rl_zero | 200/— | 31.48 |
| g1.walk_g1_dof29 | rl_zero | 200/— | 22.98 |
| h1.stand | rl_zero | 200/— | 10.85 |
| h1.walk | rl_zero | 200/— | 11.28 |
| libero.pick_bbq_sauce | replay | 62/134 | 15.16 |
| libero_90.kitchen_scene10_close_the_top_drawer_of_the_cabinet | replay | 64/71 | 11.39 |
| libero_90.kitchen_scene2_open_the_top_drawer_of_the_cabinet | replay | 59/64 | 11.79 |
| libero_90.kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle | replay | 96/119 | 12.68 |
| libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet | replay | 124/137 | 12.96 |
| libero_90.kitchen_scene5_close_the_top_drawer_of_the_cabinet | replay | 62/65 | 11.56 |
| maniskill.lift_peg_upright | replay | 164/176 | 12.87 |
| maniskill.stack_cube | replay | 118/248 | 13.73 |
| rlbench.close_box | replay | 147/247 | 11.84 |
