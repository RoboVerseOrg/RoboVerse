### 4.Migrating a New Task into RoboVerse

We support two ways to bring an external task into the RoboVerse Learn pipeline:

#### Approach 1: Direct Integration (Quick Migration)

The fastest way to integrate a new task is to:

1. **Copy the task codebase** (from an external repo) into `roboversa_learn/`
2. Replace any simulator-specific API calls with `Handler` equivalents
3. Convert raw observations into RoboVerse `TensorState` via `get_state()`
4. Move simulator-related config (e.g. robot model path, asset layout, `dt`, `decimation`, `n_substeps`) into `ScenarioCfg` and Metasim config files

This transforms the original task into a RoboVerse-compatible format while preserving its logic and structure.

**Cross-simulator support is now enabled for this task.**

####  Approach 2: Structured Wrapper Integration

To enable better reuse and cross-task comparison:

1. **Subclass `BaseTaskWrapper`**
2. Implement standardized interfaces: `_reward()`, `_observation()`, `_terminated()`
3. Use callbacks (`pre_sim_step`, `post_sim_step`, `reset_callback`) as needed
4. Leverage existing `Handler` and `ScenarioCfg` setup from Approach 1

This approach supports full compatibility with:

- **Multi-task learning benchmarks**
- **One-click algorithm switching**
- **Clean architectural separation between task, sim, and learning logic**

------

>  With either approach, you can quickly benchmark new tasks under different simulators or algorithms — with no boilerplate or duplicate integration.
