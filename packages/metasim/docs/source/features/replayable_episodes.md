# Replayable episodes

An episode recorded on one machine should replay on another. That takes three things: the full
physical state at every step in a lossless form, the names that give those numbers meaning, and the
provenance that produced them. `metasim.utils.trajectory` provides all three in one file.

## What an episode file contains

`save_episode` writes a single `.npz` (float64 arrays plus a JSON header, no pickle):

- **States** `states[t]` for `t = 0 … T` (`states[t]` is the state *before* `actions[t]`): for every
  robot and object, `root_state` `(num_envs, 13)`, and where the backend reports them `joint_pos`,
  `joint_vel`, `joint_pos_target`, `joint_vel_target`, `joint_effort_target`, `body_state`.
- **Actions** `(T, num_envs, dof)`: the tensors handed to `set_dof_targets`.
- **Names**: `joint_names` and `body_names` per entity (the handler's sorted-name order, which is what
  `get_states` / `set_states` use), the entity lists, the camera configs (intrinsics and poses travel
  with the data), and the scenario config as a dict.
- **Conventions, stated in the header**: `root_state` rows are
  `[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]` in the world frame, quaternion **wxyz**.
- **Provenance** (`Provenance`): simulator, `num_envs`, configured `dt` and the backend's resolved
  physics step (`handler.physics_dt`, a contract every backend can answer), `decimation`, simulated seconds per env step, seed, control mode, installed backend
  package versions, MetaSim version, git commit and dirty flag, every asset file the config points at
  with its size and SHA-256, Python / torch / numpy versions, platform, device, creation time.

```python
from metasim.utils.trajectory import record_episode, save_episode, load_episode, check_assets
from metasim.utils.replay import verify_episode_replay

episode = record_episode(handler, handler.get_states(mode="tensor"), actions, seed=0)
save_episode(episode, "episode.npz")

back = load_episode("episode.npz")          # validates format, shapes, names, quaternions
check_assets(back)                          # {"franka.mjcf_path": "ok" | "changed" | "missing"}
print(verify_episode_replay(handler, back)) # L0 from disk: replay the actions, compare every state
```

`verify_episode_replay` refuses to compare across a different simulator, a different `num_envs` (a
broadcast replay would compare nothing), an unknown time base, or a different env step in seconds (a
different step produces a different trajectory by construction; SuperDex folds `dt x decimation` into
its solver steps, so seconds are what is compared). When a replay drifts it names the assets whose
hashes changed and the backend packages whose versions differ. Only the asset file the backend
actually loads (`cfg.file_name(simulator)`) and `extra_resources` are hashed, so the note never blames
a file the replay did not read.

## Measured

On MuJoCo a 30-step Franka episode saved to disk and loaded back replays with a maximum deviation of
**0.0** (`metasim/test/sim/test_trajectory_replay.py`). Cross-machine replay is exact only where the
backend is deterministic; the provenance is what lets a mismatch be explained rather than guessed at.

## Data collection

`scripts/advanced/collect_demo.py` writes an `episode.npz` next to each demo's `metadata.json`
(`--episode_sidecar` / `--no-episode_sidecar`), containing the env's full state before every step
and the joint targets the handler applied, with the collection seed in the provenance. Replaying such
a file in a fresh process is how a batched-collection bug was found: settling one reset env stepped
the whole batch, and the other envs' demos silently skipped those steps. They are recorded now.

## Legacy formats

`metasim.utils.demo_util.get_traj` now recognises a trajectory file by its **content**
(`detect_traj_format`): a `roboverse.episode` file, or the legacy `{robot: [{init_state, actions,
...}]}` layout. It used to decide from the substring `v2` in the path, so moving a dataset changed how
it was parsed. Legacy files carry no provenance and no velocities; convert by replaying them once with
`record_episode` on the backend you trust.
