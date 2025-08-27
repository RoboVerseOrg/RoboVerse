# Custom Extra Observations in MetaSim (`get_extra`)

## Design Philosophy

By default, MetaSim (RoboVerse) provides standard observations such as robot joint states, velocities, and camera images. However, there are situations when you might need specific data that is **not directly included in RoboVerse's default state**. 

Extra Observation system is a way to inject user-defined observation queries into handlers without modifying the handler class itself.

## Architecture

A custom Extra Observation type is defined by inheriting `metasim.queries.base.BaseQueryType` . Let's call the inherited class `extra_qeury`.

This class has only one important method you need to overwrite **__call__()** .

When an Extra Observation is used by a Task, it will be automatically bound to the underlying handler. You can then use `self.handler` to access the handler instance within the Extra Observation class.

Everytime the `handler.get_state()` is called, the handler will call the `extra_qeury.__call__()` method of the Extra Observation type, to gather the extra observations, and return them in the return value of `handler.get_state()` .

## Possible Extra Observations

Extra Observations includes:

* Positions or velocities of specific bodies or objects.
* Sensor readings like IMU data.
* World-frame poses of specific simulation geometry (e.g., sites, bodies, frames).
* Any other types you encounter...

---

## Workflow for Using Extra Observations in Tasks

Adding custom extra observations follows these steps:

1. **Declare** the extra observations you need inside your **TaskEnv.extra_spec**.
2. **Pass** these extras to your simulation **Handler**.
3. The Handler automatically **binds** these extras, linking them to your simulation instance.
4. You then implement the procedure to access the custom data at runtime inside `extra_qeury.__call__()`. `handler.get_states()` will use `extra_qeury.__call__()` to attain extras infomation in states.

Simplified workflow diagram:

```
Task declares extras ──> Handler binds extras ──> Retrieve extras at runtime
```

---

## Step-by-Step Usage Guide

### Step 1: Declare Extras in Your Task

Implement the `_extra_spec` method inside your task definition:

```python
def _extra_spec(self) -> dict:
    return {
        "imu_pos": SitePos("imu"),   # World position of IMU sensor site
        "head_vel": BodyVel("head"), # Velocity of the robot's head
    }
```

* The **keys** (e.g., `"imu_pos"`) are user-defined and identify the returned data.
* The **values** (`SitePos`, `BodyVel`) are pre-defined QueryTypes that fetch specific data from your simulator.

Common built-in queries:

* `SitePos("site_name")`: Returns world-frame position.
* `BodyVel("body_name")`: Returns velocity of a body.

---

### Step 2: Pass Extras to the Handler

When creating your simulation Handler, pass your extras dictionary (`_extra_spec`) via the `optional_queries` parameter:

```python
task = CrawlEnv(CrawlEnv.scenario)
extra_queries = task._extra_spec()

handler = YourSimHandler(
    scenario=task.scenario,
    optional_queries=extra_queries
)

handler.launch()  # Extras get bound here
```

---

### Step 3: Access Extra Observations at Runtime

While your simulation is running, simply call:

```python
extras = handler.get_extra()
```

This returns a dictionary containing your custom observations:

```python
{
    "imu_pos": tensor([[0.1, 0.2, 0.3], [...]]),
    "head_vel": tensor([[0.01, 0.02, 0.03], [...]])
}
```

You can directly use this data for:

* Reward computations
* Logging and analysis
* Visualization purposes

Additionally, when you call:

```python
state = handler.get_state()
```

the returned `state` dictionary **automatically includes all the extra observations you've defined**. This way, your extras seamlessly integrate into RoboVerse's default simulation state.

---

## Adding Your Own Custom Query

If the existing queries in MetaSim do not fit your needs, you can define a custom query type by subclassing `BaseQueryType`.

Each custom query involves two steps:

### ① Define a subclass of `BaseQueryType`

First, create a new Python class that inherits from `BaseQueryType`. You will typically implement two methods in this class:

* **`bind_handler(self, handler)`**:
  This method is called **once** when the simulation launches. Here you save necessary references (such as simulator handles, indexes, IDs, or configurations) from your specific simulation handler.

* **`__call__(self)`**:
  This method is invoked **at runtime** every time you request extras. In this method, you use the previously stored references to retrieve the real-time data you need.

Example:

```python
from metasim.queries.base import BaseQueryType
import torch

class BodyMassQuery(BaseQueryType):
    def bind_handler(self, handler):
        # Store the handler for later use
        self.handler = handler

        # Example: simulator-specific logic
        mod = handler.__class__.__module__
        if mod.startswith("metasim.sim.mujoco"):
            self.mass = handler.physics.model.body_mass  # Example MuJoCo API call
        else:
            raise ValueError(f"Unsupported handler: {mod}")

    def __call__(self):
        # Return the stored body mass as a tensor
        return torch.tensor(self.mass)
```

### ② Use your custom query inside `_extra_spec`

After defining your custom query class, simply instantiate and use it directly in your task's `_extra_spec` method:

```python
def _extra_spec(self) -> dict:
    return {
        "body_mass": BodyMassQuery(), # Your custom query
        "imu_pos": SitePos("imu"),
    }
```

After following these steps, your handler will automatically bind your custom query, and you can seamlessly retrieve this data during runtime.

---

## Practical Tips

* **Always ensure** simulator-specific code in your custom query is correctly placed inside the `bind_handler` method. This ensures efficient performance, as binding happens once at simulation launch rather than repeatedly at runtime.

* Your `__call__` method should be fast and efficient, simply retrieving pre-stored references or quickly fetching real-time data.

* It’s good practice to return data consistently as PyTorch tensors (or similarly structured arrays), as these integrate well with downstream RL pipelines.

---

Now you're fully equipped to flexibly extend your RoboVerse simulation with any custom extra observations you need!
