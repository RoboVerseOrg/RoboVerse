# MetaSim for RL

## Layers

```{mermaid}
graph TD
    A[0 Simulator]
    B[1 Handler]
    C[2 Gym.Env]
    D[3 RL Framework or Benchmark]

    A --> B
    B --> C
    B --> D
    C --> D
```

### Layer 0: Simulator
**DON'T USE!**

### Layer 1: Handler
Corresponds to `env.handler` in MetaSim

Interface:
- `get_observation()`
- `get_reward()`
- `get_success()`
- `get_time_out()`
- ...

Current implementation:
- v1 **[Feishi]**
- v2 **[Boshi]**

### Layer 2: Gym.Env
Corresponds to `env` in MetaSim

Layer 2 is a light-weight wrapper of Layer 1.

Interface:
- `reset()`
- `step()`
- `render()`
- `close()`

Current implementation:
- 2.1 Env: (corresponds to `env`, don't support `Gym.Env`!) **[Deprecated?]**
- 2.2 Gym.Env 0.26: specialized for HumanoidBench **[Haozhe,Yutong]**
    - [x] MuJoCo **[Haozhe,Yutong]**
    - [x] IsaacGym **[Yutong]**
- 2.3 Chaoyi's Gym.Env if necessary **[Chaoyi]**
- 2.4 Gym.Env 1.0: merge all above implementations, final goal **[TODO]**

### Layer 3: RL Framework
- 3.1 RSL_RL's VecEnv + RSL_RL **[Chaoyi]**
- 3.2 StableBaseline3 integration **[Yutong]**

### Layer 4: RL Tasks on exising benchmarks
Interface: depends on the specific RL framework or benchmark

Current implementation:
- 4.1 HumanoidBench: **[Haozhe,Yutong]**

## TODOs
- Layer 1 Handler
  - [ ] v2: Boshi implement metasim's handler using `get_states()` and `set_states()` as core functions.
  - [ ] v1:
    - [ ] Feishi and Charlie ensure interface is aligned across IsaacSim and MuJoCo.
    - [ ] Feishi support new handler properties (num_envs, num_obs, num_actions. Any else?)
- Layer 2: Env
  - [ ] Serve for Layer 3, everyone can implement their own Gym.Env for current stage.
  - [ ] (Optional) We will finally merge all above implementations into one and support Gym.Env 1.0.
- Layer 3: RL Framework or Benchmark
  - This layer is based on layer 1 or 2 (except 2.1, which is deprecated!), but not based on layer 0. In this way, cross-simulator is guaranteed.
  - [ ] Yutong and Haozhe implement HumanoidBench
  - [ ] Chaoyi implement RSL_RL
    - [ ] Chaoyi will start from PickCube + IsaacLab Handler

## Get reward from states (TODO: need update?)
```python
states = env.handler.get_states()
```

`states` is a list of state of each environment. For Mujoco, it has single elements.

The structure of a single state is as follows:
```text
{
    "{object_name}": {
        "pos": [x, y, z],
        "rot": [w, x, y, z],
        "vel": [x, y, z],
        "ang_vel": [x, y, z],
        // below are optional fields for articulated objects
        "dof_pos": {
            "{joint_name}": float,
            ...
        },
        "dof_vel": {
            "{joint_name}": float,
            ...
        },
        // below are optional fields for articulated objects that have actuators
        "dof_pos_target": {
            "{joint_name}": float,
            ...
        },
        "dof_vel_target": {
            "{joint_name}": float,
            ...
        },
        "dof_torque": {
            "{joint_name}": float,
            ...
        }
    },
    // bodies are part of the articulated objects linked by joints
    "metasim_body_{body_name}": {  // the prefix "metasim_body_" is for compatibility with the old code
        "pos": [x, y, z],
        "rot": [w, x, y, z],
        "vel": [x, y, z],
        "ang_vel": [x, y, z],
        "com": [x, y, z],
        "com_vel": [x, y, z],
    },
    // sites are defined in task cfg, by the base (either a object root or a body root) and the relative pose
    "metasim_site_{site_name}": {  // the prefix "metasim_site_" is for compatibility with the old code
        "pos": [x, y, z],
        "rot": [w, x, y, z],
        "vel": [x, y, z], # Optional, only valid if sensor data is present
        "ang_vel": [x, y, z], # Optional, only valid if sensor data is present
    }
}
```
