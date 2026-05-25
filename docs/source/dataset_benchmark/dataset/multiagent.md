# Multi-Agent (Bimanual) Datasets

RoboVerse's trajectory format is multi-agent native. A dataset file stores one
entry **per agent, keyed by robot name** — the same on-disk layout single-agent
datasets already use. A single-agent file is therefore just the one-key special
case, so existing datasets keep working unchanged.

This is what makes bimanual workflows (two independent arms acting
simultaneously, e.g. ManiSkill's `TwoRobotStackCube-v1` style tasks) expressible
without inventing a parallel format.

## On-disk format

A `*_v2.pkl` file is a dict keyed by robot name. Each agent maps to a list of
demos; each demo carries `init_state`, `actions`, and optional `states`:

```python
{
    "franka_left":  [{"init_state": {...}, "actions": [...], "states": None}, ...],
    "franka_right": [{"init_state": {...}, "actions": [...], "states": None}, ...],
    "metadata": {"num_agents": 2, "agents": ["franka_left", "franka_right"]},
}
```

Each agent's `init_state` lists that agent's robot entry plus any **shared
objects** (the cube both arms coordinate around). Per-agent actions are
namespaced as `{"dof_pos_target": {...}}`.

## Loading with `get_traj`

The canonical loader `metasim.utils.demo_util.get_traj` takes either a single
robot (single-agent, unchanged) or a **list of robots** (multi-agent):

```python
from metasim.utils.demo_util import get_traj

robots = [franka.replace(name=n) for n in ["franka_left", "franka_right"]]
init_states, all_actions, all_states = get_traj("bimanual_handover_v2.pkl", robots)
```

Passing the list returns the **same three-tuple shape** as the single-agent
path, with every agent merged into each per-step dict:

- `init_states[d]["robots"]` holds every arm; `init_states[d]["objects"]` holds
  the shared objects once.
- `all_actions[d][t]` is `{robot_name: {"dof_pos_target": ...}}` for **all**
  agents at step `t` — exactly what `handler.set_dof_targets([...])` consumes.
- `all_states[d][t]` unions each agent's `robots`/`objects` (or is `None` for
  action-only demos).

Because the shape is identical, the same replay / collection code paths drive
one arm or many. Multi-agent loading requires the v3 namespaced format
(`v2_as_v3=True`, the default); `v2_as_v3=False` with a robot list raises, since
namespacing is what keeps each agent's actions indexed by name.

## Runnable example

`get_started/8_multiagent_dataset.py` builds a coordinated two-Franka handover
trajectory, saves it as a real `*_v2.pkl`, loads it back through `get_traj`, and
replays both arms simultaneously to video:

```bash
MUJOCO_GL=egl python get_started/8_multiagent_dataset.py --sim mujoco
```

## Single-embodiment bimanual vs. two agents

Two distinct cases share this format:

- **Single-embodiment bimanual** (one URDF with two arms, e.g. ALOHA / RoboTwin
  AgileX) — one robot entry whose action dict spans all joints. See the
  [RoboTwin Integration](../integrations/robotwin.md).
- **Two independent agents** (two separate robot entities) — the case above,
  one keyed entry per agent.
