# RoboVerse Learn  Overeview

RoboVerse Learn consists of Task Wrappers and Learning Framework.  
Its goal is to present *one* standard interface that:

* Lets any algorithm (PPO, SAC, BC, etc.) work with any task  
* Hides simulator & task differences, so you can swap tasks, simulators or algorithms with minimal friction  

---

##  Design Principles

| #     | Principle                                | Key Points                                                   |
| ----- | ---------------------------------------- | ------------------------------------------------------------ |
| **1** | **Standardised Wrapper API**             | • `TaskWrapper` exposes `step / reset / _reward / _observation / _success`.<br>• Once an algorithm is connected to a single `TaskWrapper`, it can seamlessly switch to any other task simply by replacing the wrapper.<br/>• Upper‑level algorithms need not care whether the backend is MuJoCo, Isaac, etc. |
| **2** | **Minimise Task‑Migration Cost**         | • Add a task: just subclass / compose a wrapper.<br>• Switch simulator: wrappers/algorithms stay unchanged.<br>• Directory layout, Configs management（except the sim-related part）, training scripts all stay the same. |
| **3** | **Reusable Reward & Checker Primitives** | • Tasks build complex logic by *composing* primitives → no copy‑paste across tasks. |

---

## 1. Module Composition

| Sub‑module              | Responsibilities                                             |
| ----------------------- | ------------------------------------------------------------ |
| **Task Wrapper**        | • Combines a `Handler` & exposes `step / reset`.<br>• Assembles Reward / Observation / Success .<br>• Provides `pre_sim_step` & `post_sim_step` callbacks for *task‑level* DR. |
| **Handler (Metasim)**   | • `set_state / get_state / get_extras` unified across engines.<br>• *Physics‑level* DR (`pre_sim_step`).<br>• Pure simulator adapter—no algorithm logic. |
| **Learning Framework**  | • Any RL / IL algorithm.<br>• No simulator knowledge.        |
| **Custom Util Wrapper** | • Provide lightweight extensions (e.g., NumPy-to-Torch conversion, first-frame caching) to support logging, preprocessing, or offline data collection without modifying core task logic. |

---

## 2. Interface List

| Method                        | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `step(action)`                | Runs one simulation step: calls `pre_sim_step`, then `handler.simulate()`, then `post_sim_step`; returns `(obs, reward, done, info)` |
| `reset()`                     | Resets the environment and applies `reset_callback`, returns initial observation |
| `pre_sim_step()`              | (Optional) Hook for task-level domain randomization before simulation |
| `post_sim_step()`             | (Optional) Hook for post-processing (e.g., observation noise) |
| `get_state()` / `set_state()` | Unified simulator-agnostic state interface using `TensorState` |
| `get_extras(spec)`            | Returns task-specific quantities (e.g., site poses, contact forces) via query descriptors |

## 3. Domain Randomisation Layers

| Layer             | Location                      | Examples                                             |
| ----------------- | ----------------------------- | ---------------------------------------------------- |
| **Physics‑level** | `Handler`                     | Friction, mass, light, material                      |
| **Task‑level**    | `Wrapper.pre/post_sim_step()` | Action noise, observation noise, initial‑pose jitter |

*Rule:* Simulator parameters → Handler; task‑coupled noise → Wrapper.

------

## 4.Migrating a New Task into RoboVerse

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