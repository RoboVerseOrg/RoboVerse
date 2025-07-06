# Collect Demonstrations 

This tutorial explains how to use `metasim/scripts/collect_demo.py` to collect expert demonstrations for imitation learning in RoboVerse.

---

## 📌 Overview

`collect_demo.py` is a utility script that replays pre-defined trajectories (from `traj_filepath`) and records observations, actions, and metadata into a structured dataset. The collected demos are used for training behavior cloning or other imitation learning models.

---

## ⚙️ 1. Basic Usage

### Example
To collect demonstrations for `pick_cube` task, run the following command:

```bash
python collect_demo.py \
  --task pick_cube \
  --robot franka \
  --sim isaaclab \
  --num_envs 2 \
  --headless True \
  --run_unfinished
```

> **Important:** You must include **exactly one** of the following collection modes:
>
> - `--run_all`: Collect all demos from scratch (overwrite existing ones)
> - `--run_unfinished`: Only collect demos that are missing or incomplete ✅ recommended
> - `--run_failed`: Retry demos that previously failed

---

## Argument Reference

| Argument | Description |
|----------|-------------|
| `--task` | Task name (e.g., `pick_cube`, `plug_charger`) |
| `--robot` | Robot platform (e.g., `franka`) |
| `--sim` | Simulator backend (`isaaclab`, `mujoco`, etc.) |
| `--run_all` / `--run_unfinished` / `--run_failed` | Choose exactly one |
| `--num_envs` | Number of parallel environments |
| `--headless` | Disable rendering window for faster collection |

---

## 📁 2. Where is data saved?

Demo data is stored in:

```
roboverse_demo/demo_<sim>/<TaskName>-Level<level>[-<cust_name>]/robot-<robot>/demo_XXXX/
```

Each folder contains:
- A sequence of observations and actions
- `status.txt` indicating `success` or `failed`

You can use `--cust_name` to customize the folder name:

```bash
--cust_name my_experiment_name
```

---

## 🎛️ 3. Advanced Options

| Argument | Description |
|----------|-------------|
| `--demo_start_idx` | Index of the first demo to collect |
| `--max_demo_idx` | Maximum demo index to collect |
| `--retry_num` | Number of times to retry a failed demo |
| `--tot_steps_after_success` | Extra steps to collect after success (default: 20) |
| `--table` | Whether to add a table object in the scene |
| `--scene` | Optional scene name |
| `--render.*` | Render config (camera, resolution, etc.) |
| `--random.*` | Domain randomization config |

Example:
```bash
python collect_demo.py \
  --task plug_charger \
  --run_all True \
  --num_envs 4 \
  --cust_name vision_exp \
  --retry_num 2 \
  --headless True \
  --tot_steps_after_success 20
```

---

## 🧠 4. Notes

- The script reads pre-recorded trajectories via `task.traj_filepath`
- If a demo fails N times (`retry_num`), it is marked as failed and skipped
- Successful demos are skipped unless `--run_all` is set
- Parallelism helps speed up collection but increases GPU load
- Currently uses global variables for tracking progress (e.g. `tot_success`) — future refactor may move to a `ProgressManager`

---

## 🧼 5. Cleanup and Inspection

- Demos that have a `status.txt` file with the content `failed` will be retried automatically when using `--run_failed`. There is no need to delete these folders manually.
- You can visualize collected observations or use them for training downstream agents.


---

## ✅ Summary

- Use `collect_demo.py` to automate demonstration collection.
- Start with `--run_unfinished` or `--run_all`.
- Use `--cust_name` to organize your datasets.
- Ensure `traj_filepath` is valid in your task config.

---
