# Hybrid Rendering (MuJoCo physics, Isaac Sim renderer)

`HybridSimHandler` runs physics in one backend and rendering in another. The canonical pairing is
MuJoCo for physics (one CPU worker process per env) and Isaac Sim for rendering (one RTX stage that
holds all envs). After every step the physics state is pushed into the renderer, so the frames show
exactly the poses the physics backend computed.

Use it when you want MuJoCo's speed, determinism, or contact model for the dynamics and Isaac Sim's
ray-traced RGB / depth for the observations, or when you need many rendered envs without paying for
PhysX on each of them.

---

## How it works

```
set_dof_targets / simulate            get_states
        │                                  │
        ▼                                  ▼
 physics_handler (mujoco, N workers)  render_handler (isaacsim, N envs on one stage)
        │  TensorState (cpu)                ▲
        └── state_to_device(renderer) ──────┘   cameras come from the renderer,
                                                robots/objects from physics
```

- `HybridSimHandler.simulate()` steps physics, then calls `render_handler.set_states(...)` with the
  physics state moved to the renderer's device (`metasim.utils.state.state_to_device`).
- `HybridSimHandler.get_states()` returns robots and objects from physics and cameras from the
  renderer, so downstream code sees one ordinary `TensorState`.
- The physics scenario should be **camera-free and headless**: physics never renders, and MuJoCo
  only skips its GL renderer when both hold. Build it with `scenario.replace(cameras=[], headless=True)`.

## Usage

```python
from metasim.sim import HybridSimHandler
from metasim.utils.setup_util import get_sim_handler_class
from metasim.constants import SimType

scenario = ...                                            # cameras defined here
physics = get_sim_handler_class(SimType.MUJOCO)(scenario.replace(simulator="mujoco", cameras=[], headless=True))
renderer = get_sim_handler_class(SimType.ISAACSIM)(scenario.replace(simulator="isaacsim"))
handler = HybridSimHandler(scenario, physics, renderer)
handler.launch()                                          # starts Kit for the renderer

handler.set_dof_targets(actions)
handler.simulate()
state = handler.get_states(mode="tensor")                 # state.cameras["cam"].rgb: (N, H, W, 3) uint8
```

If the process already hosts an Isaac Sim application (tests, notebooks, an outer `AppLauncher`),
pass it through: `handler.launch(simulation_app=app)`. Starting a second `AppLauncher` in one
process shuts the first one down.

The end-to-end demo is `examples/5_hybrid_sim.py`:

```bash
python examples/5_hybrid_sim.py --sim mujoco --renderer isaacsim --headless   # writes examples/output/5_hybrid_sim_mujoco_render_isaacsim.mp4
```

## Installation

Rendering needs Isaac Sim **and** Isaac Lab in the same environment. The pip `isaacsim` wheel does
not bring Isaac Lab, so it is installed from source; the exact, verified sequence (Python 3.11,
Isaac Sim 5.0, Isaac Lab 2.2.1, CUDA 12.8 torch on Blackwell GPUs) is scripted in
`tools/install/isaacsim5.sh` (documented in `packages/metasim/requirements/isaacsim5.txt`). `python -m metasim doctor --backend isaacsim` checks
the result against the tested versions in `metasim/sim/_versions.py`.

## Verification

`metasim/test/sim/test_hybrid_render.py` (marker `sim("isaacsim")`, skipped without Isaac Sim)
asserts on a 4-env hybrid that:

- object poses and robot joint positions in the renderer equal the physics state to `1e-5` in every env;
- `rgb` is `(N, H, W, 3)` uint8 and `depth` is `(N, H, W)`, both from the renderer;
- every env renders a non-flat frame and envs driven to different targets render different frames;
- one deferred-flush render pass already shows a teleported object (matches a two-pass refresh);
- the wrapped physics / render handlers' own `get_states` reflect the hybrid step (cache invalidation).

## Performance

Measured on one RTX 5090, Isaac Sim 5.0, Isaac Lab 2.2.1, MuJoCo 3.x, Franka + two primitives,
one 256×256 pinhole camera per env, headless. "simulate" includes the physics step and the state
push into the renderer; "get_states" includes the RTX render and the readback.

| envs | physics only (ms/step) | simulate incl. sync + render (ms/step) | get_states readback (ms) | total per env (ms) | GPU memory (GB) |
|-----:|-----------------------:|---------------------------------------:|-------------------------:|-------------------:|----------------:|
|    1 |                   0.45 |                                   11.0 |                      2.3 |               13.3 |             5.8 |
|    4 |                   1.02 |                                   39.8 |                     29.9 |               17.4 |            15.8 |
|    8 |                   1.98 |                                   85.0 |                     38.7 |               15.5 |            25.8 |
|   16 |                      – |                                      – |                        – |                  – | out of memory (both 256² and 128²) |

Physics is negligible; the step is the RTX render (one render product per env camera) and the
`get_states` cost is the sensor readback of those products. Numbers are a 20-step mean of one run;
RTX timings vary by tens of percent between runs, so read the table for orders of magnitude. The sync itself is one deferred-flush
`set_states` plus a single render pass: on one env it costs 9.3 ms where the previous
`set_states` (two passes) followed by `refresh_render` (two more) cost 22.3 ms for the same frame.

GPU memory, not time, is the limit: each env adds roughly 3 GB of render-product and stage memory,
so 16 envs do not fit on a 32 GB card even at 128×128 (the run drowns in Vulkan
`ERROR_OUT_OF_DEVICE_MEMORY` and has to be killed). Size the stage to ~8 envs per 32 GB GPU and add
processes/GPUs beyond that; the per-env figure keeps falling up to that point.
