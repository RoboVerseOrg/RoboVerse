# Task System

## Architecture and Philosophy

A **task** is a wrapper that contains gym APIs on top of a Handler. In our RoboVerse design philosophy, we put all simulation related contents into a scenario config and instantiated with a handler. All other contents on top of the simulation (reward, observation, etc.) are implemented with different layers wrappers.

A task is instantiated with two parameters: **scenario** and **device**. The **scenario** is a scenario config that specifies the underlying simulated environment, and device is used to specify which device this scenario is instantiated on. In the initialization, the scenario will be instantiated into a handler.

When defining your own task, you need to inherit from `BaseTaskEnv` and implement multiple methods including `_observation`, `_privileged_observation`, `_reward`, `_terminated`, `_time_out`, `_observation_space`, `_action_space`, `_extra_spec`. These methods are basic building blocks of a task with gym-style APIs.

## Migrating a New Task into RoboVerse

We encourage two ways to bring an external task into the RoboVerse Learn pipeline:

### Approach 1: Direct Integration (Quick Migration)

The fastest way to integrate a new task is to:

1. **Copy the task codebase** (from an external repo) into `roboversa_learn/`
2. Replace any simulator-specific API calls with `Handler` equivalents
3. Convert raw observations into RoboVerse `TensorState` via `get_state()`
4. Move simulator-related config (e.g. robot model path, asset layout, `dt`, `decimation`, `n_substeps`) into `ScenarioCfg` and Metasim config files

This transforms the original task into a RoboVerse-compatible format while preserving its logic and structure.

**Cross-simulator support is now enabled for this task.**

###  Approach 2: Structured Wrapper Integration

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
