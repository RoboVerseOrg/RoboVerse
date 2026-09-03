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

## Runnable examples

`examples/8_multiagent_dataset.py` builds a coordinated two-Franka handover
trajectory, saves it as a real `*_v2.pkl`, loads it back through `get_traj`, and
replays both arms simultaneously to video:

```bash
MUJOCO_GL=egl python examples/8_multiagent_dataset.py --sim mujoco
```

The same trajectory is also exposed as a registered task, so it replays through
the **canonical pipeline** (`scripts/advanced/replay_demo.py`) — which now passes
the full robot list to `get_traj` whenever a task declares more than one robot:

```bash
MUJOCO_GL=egl python scripts/advanced/replay_demo.py \
    --task bimanual.franka_handover --sim mujoco --headless
```

`examples/9_maniskill_two_robot_stack_cube.py` does the same round trip with
**real ManiSkill data**: it fetches the official `TwoRobotStackCube-v1`
demonstrations, converts one episode into the name-keyed `*_v2` format, loads
both Panda arms through `get_traj`, and replays the recorded states on MuJoCo:

```bash
MUJOCO_GL=egl python examples/9_maniskill_two_robot_stack_cube.py --sim mujoco
```

The ManiSkill `.h5` stores one articulation per agent
(`panda_wristcam-agent-0` / `-agent-1`) plus the shared cubes; converting it is
just a regrouping into one keyed entry per agent. Replay uses the recorded
**states** (kinematic playback) rather than open-loop action targets: the demos
were collected under SAPIEN's `pd_joint_delta_pos` controller, and closed-loop
contact dynamics do not transfer across simulators, so state replay is the
faithful cross-sim view of the dataset.

## Single-embodiment bimanual vs. two agents

Two distinct cases share this format:

- **Single-embodiment bimanual** (one URDF with two arms, e.g. ALOHA / RoboTwin
  AgileX) — one robot entry whose action dict spans all joints. See the
  [RoboTwin Integration](../integrations/robotwin.md).
- **Two independent agents** (two separate robot entities) — the case above,
  one keyed entry per agent.

The single-embodiment bimanual case is demonstrated by
`examples/10_robotwin_aloha_replay.py`, but note:

```{warning}
`examples/10_robotwin_aloha_replay.py` is **experimental** and **not
out-of-the-box** (unlike examples 8 and 9, which run from a clean MuJoCo
install). It needs a local RoboTwin clone, its ~3.74 GB asset pack, a separate
`robotwin` conda env, and a curobo build for the local GPU arch to collect a
bridge pickle first. The manipulated object is rendered as a primitive-cube
proxy (not the real mesh), and only joint *motion* — not task *success* — has
been confirmed. Treat it as a data-bridge demo, not a benchmark.
```
