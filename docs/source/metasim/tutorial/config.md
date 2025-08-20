
# Configuration System

## Design Philosophy

A **scenario config** is a simulator-agnostic configuration class that defines a simulation. It contains defnitions for objects, robots, lightings, physics parameters and so on. All these definitions will be instantiated into a real simulation when passed to a handler.

## System Organization

All simulator-agnostic static configurations for a simulation is organized in a configuration system. The root of the configuration is a **scenario config**. It contains other configurations:

```python
@configclass
class ScenarioCfg:
  scene: SceneCfg | None = None
  robots: list[RobotCfg] = []
  lights: list[BaseLightCfg] = [DistantLightCfg()]
  objects: list[BaseObjCfg] = []
  cameras: list[BaseCameraCfg] = []
  render: RenderCfg = RenderCfg()
  sim_params: SimParamCfg = SimParamCfg()
  simulator: str | None = None
  renderer: str | None = None
  num_envs: int = 1
  headless: bool = False
  env_spacing: float = 1.0
  decimation: int = 25
```
