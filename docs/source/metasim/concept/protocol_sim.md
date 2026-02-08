# Protocol Simulation

## Overview

Protocol simulation emulates a robot communication stack (for example, Unitree SDK2 DDS) against a simulator-backed robot.

This lets you run an existing real-robot controller unchanged, while replacing the robot hardware with a MetaSim robot instance.

Typical use cases:

- Validate a deployment controller before running on hardware.
- Debug command/state mismatches between simulator and real stack.
- Reproduce bring-up issues (startup, standby/takeover, torque limits) in a safer loop.

## Package Structure

Protocol simulation is split into two layers:

- Core framework (vendor-agnostic): `metasim/protocol_sim/core`
- Concrete protocol plugins (vendor-specific): `roboverse_pack/protocol_sim/protocols`

Current concrete plugin:

- Unitree SDK2 plugin: `roboverse_pack/protocol_sim/protocols/unitree_sdk2`

Current CLI entrypoint:

- `scripts/protocol_sim/unitree_sdk2_sim_server.py`

## Data Flow

Each control step in `RobotProtocolServer` follows this sequence:

1. Read observation from simulator (`MetaSimAdapter.read_observation`).
2. Pull latest inbound protocol command (`Transport.get_latest_command_with_token`).
3. Decode command into canonical representation (`ProtocolCodec.decode_command`).
4. Choose standby or protocol control path (based on standby config and command activity).
5. Compute torque/effort (`ActuationModel.compute_effort` or `StandbyController.compute_effort`).
6. Apply effort to simulator (`MetaSimAdapter.apply_effort`), then step simulator.
7. Encode outbound state messages (`ProtocolCodec.encode_messages`) and publish by channel.

## Core Interfaces

Core interface definitions live in `metasim/protocol_sim/core/interfaces.py`:

- `Transport`: messaging I/O boundary (DDS/ROS/LCM/etc.)
- `ProtocolCodec`: semantic conversion between vendor messages and canonical command/state
- `ActuationModel`: canonical command -> simulator effort
- `StandbyController`: safe controller before external controller takeover
- `ExternalAssist`: optional safety assist (for example, elastic-band force)

Core server and adapter:

- `metasim/protocol_sim/core/server.py`
- `metasim/protocol_sim/core/sim_adapter.py`

Canonical data models:

- `metasim/protocol_sim/core/types.py`
  - `CanonicalRobotCommand`
  - `SimRobotObservation`

## Unitree SDK2 Plugin

Unitree-specific implementation is intentionally kept out of `metasim` core:

- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/transport.py`
- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/codec.py`
- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/actuation.py`
- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/standby_policy.py`
- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/server.py`

The plugin depends on `unitree_sdk2py` for DDS transport and IDL message types.

Supported robot profiles are defined in:

- `roboverse_pack/protocol_sim/protocols/unitree_sdk2/spec_registry.py`

At the time of writing, the registry includes `g1_dof29`.

## Quick Start

Run the Unitree SDK2 simulation server:

```bash
python scripts/protocol_sim/unitree_sdk2_sim_server.py \
  --sim mujoco \
  --robot g1_dof29 \
  --domain-id 1 \
  --iface lo \
  --auto-remote \
  --elastic-band \
  --elastic-band-height 2.5 \
  --elastic-band-length 0.1
```

Then run the controller (unchanged) against the same DDS domain:

```bash
python scripts/unitree_deploy/deploy_real.py lo g1_dof29.yaml --domain-id 1
```

## Important Runtime Options

Main options from `scripts/protocol_sim/unitree_sdk2_sim_server.py`:

- `--match-task`: align dt/scenario/initial state to a task (`registry_key` or `module:TaskClass`)
- `--actuation-gains {cmd,robot_cfg}`: use gains from incoming LowCmd or from RobotCfg
- `--no-standby`: disable standby controller
- `--standby-cmd-timeout`: revert to standby after command stream timeout
- `--elastic-band`: enable external safety assist
- `--elastic-band-height`, `--elastic-band-length`: anchor and rest length tuning

## Live Elastic-Band Control

When `--elastic-band` is enabled and the server runs in an interactive terminal (TTY),
you can tune assist strength live without restarting.

Key mode (single-key input):

- `7`: stronger assist (shorter rest length)
- `8`: weaker assist (longer rest length)
- `]`: move anchor up
- `[`: move anchor down
- `r`: release elastic band immediately
- `s`: print current height/length
- `h`: print help

Line mode commands (fallback when key mode is unavailable):

- `length <m>`
- `height <m>`
- `release` (immediate)
- `show`
- `help`

Notes:

- Live tuning is disabled when stdin is not a TTY.
- Adjustment increment for `7/8/[ ]` is controlled by `--elastic-band-key-step` (meters).
- Rest length is clamped to `>= 0`.

## Programmatic Usage

```python
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.server import (
    UnitreeServerArgs,
    build_unitree_sdk2_server,
)

args = UnitreeServerArgs(
    sim="mujoco",
    robot="g1_dof29",
    domain_id=1,
    iface="lo",
    auto_remote=True,
    standby=True,
)

server = build_unitree_sdk2_server(args)
server.start()
try:
    server.run_forever()
finally:
    server.close()
```

## Extending to New Protocols

To add another vendor protocol:

1. Create a new plugin package under `roboverse_pack/protocol_sim/protocols/<vendor>`.
2. Implement `Transport`, `ProtocolCodec`, and `ActuationModel`.
3. Optionally implement `StandbyController` and/or `ExternalAssist`.
4. Assemble these pieces into `RobotProtocolServer`.
5. Add a script entrypoint under `scripts/protocol_sim/`.

Keep dependency direction one-way:

- Plugin code may import `metasim.protocol_sim.core`.
- Core code must not import plugin-specific modules.
