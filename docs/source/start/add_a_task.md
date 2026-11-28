# Add a task

A task is registered only after it passes the gate. The gate is not a review step you negotiate with;
it is what the word "task" means here. This page is the whole path from a file to a green row.

## What a task is

One module under `roboverse_pack/tasks/<family>/`, one class, decorated once:

- a class-level `scenario` (`ScenarioCfg`: robots, objects with asset paths that exist, optional cameras);
- a checker (`metasim.task.checkers`: position shift, relative bounding box, joint position, site
  predicates) or, for a dense-reward RL task, `_reward`;
- a `prompt_bank` (canonical instruction plus train and held-out paraphrases);
- a demo file (`traj_filepath`, a `roboverse_data/trajs/...` path) or an explicit `no_demos = True`;
- `supported_simulators` when the assets exist for some backends only.

Start from `roboverse_pack/tasks/task_template.py`. Variants are parameters or instances, never
generated classes: a module may register at most 16 classes (`tests/test_task_contract_static.py`).

## 1. Assets that exist

Every `roboverse_data/...` path you write must exist on the Hub dataset. The pull-request check
`tools/assets/check_manifest.py --hub-only` fails otherwise; run it locally first:

```bash
python tools/assets/check_manifest.py --paths roboverse_pack/tasks/<family>/<task>.py
```

Declare only the formats you ship (an MJCF for MuJoCo, a USD for Isaac). A path to a file that does
not exist is not "support for that backend", it is a broken config.

## 2. Demos without a human

Write a scripted plan over the skill library (`metasim.bench.skills`: `pick`, `place_on`, `push_to`,
`stand_upright`, or `MoveTo` / `Gripper` primitives with your own targets), register it in
`TABLETOP_PLANS` or pass it as the policy, and record:

```bash
python -m metasim bench record --task <name> --sim mujoco --episodes 120 \
    --policy metasim.bench.skills:ScriptedPolicy --out demos/<name>/v2/franka_v2.pkl.gz
```

Only the episodes the task scored as successful are kept, so the recorder also tells you whether the
checker agrees with the plan. A trained policy or a served model records the same way.

## 3. The gate

```bash
python -m metasim bench health --sim mujoco --task <name> --demos demos/<name>/v2 --episodes 20 --floor --repeat 2
```

Green means: the demos replay at 0.80 or better on the backend they were recorded on, the hold policy
(no motion) scores exactly 0, two replays of the same instances agree, and every initial state keeps each
robot joint inside its limits. `low`, `unreliable`, `flaky`, `invalid` and `error` are each a different bug,
named in the row. Fix the task, not the threshold. (`invalid` caught the shipped `lift_peg_upright` demos
on 2026-11-28: the fingers start at 1.0 in a 0 to 0.04 range, and the gate and the IL collector resolved
that differently, so the same demos replayed at 1.0 in one and 0.45 in the other.)

## 4. Publish

Upload the demo file to the Hub under `trajs/<family>/<name>/v2/`, set `traj_filepath`, open the
pull request. CI runs the static contract and the manifest check; the nightly gate runs the replay on
MuJoCo and Isaac Sim 6 and publishes the table. A task that is red two nights in a row is unregistered
by the next pull request and listed in `roboverse_pack/tasks/RETIRED_TASKS.md` with the commit to
restore it from.

## Bringing a retired task back

`RETIRED_TASKS.md` names the file and the commit. Restore it, make it pass the gate, open the pull
request. The 3021 classes retired on 2026-11-27 are not lost; they are waiting for exactly this.
