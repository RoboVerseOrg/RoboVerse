# Protocol Simulation

Protocol simulation lets an external controller run asynchronously against a MetaSim simulator. The core package provides only the orchestration layer; concrete message formats, network stacks, and controller-specific behavior belong in downstream packages.

## Core Flow

Each protocol tick follows this sequence:

1. Read the current MetaSim tensor state.
2. Sample the latest inbound command from a transport.
3. Decode the command only when the transport reports a new command token.
4. Convert the latest decoded command into a MetaSim action, or no action if the command should not change targets.
5. Apply the action when one is returned, step the simulator, and advance protocol time.
6. Encode outbound state messages and publish them through the transport.

If the controller sends commands more slowly than the simulator steps, the server reuses the latest decoded command on later simulator ticks.

## Extension Points

`Transport` owns asynchronous I/O. Implementations can use any communication mechanism, as long as they expose the latest command and publish outbound messages by channel. Transports that reuse a mutable message object should override `get_latest_command_with_token()` with a monotonic sequence token.

`ProtocolCodec` converts between transport messages and decoded protocol objects. The core does not prescribe a wire format or command schema.

`CommandAdapter` converts a decoded command plus current `TensorState` into MetaSim `CompatActionInput` or `None`. Returning `None` lets the server step the simulator without applying new targets. This is where downstream packages choose position, velocity, effort, mixed control semantics, or no-action behavior.

`MetaSimProtocolAdapter` wraps an existing MetaSim handler by delegating to `get_states(mode="tensor")`, `set_dof_targets()`, `simulate()`, and `close()`.

## Out Of Scope

The standalone core intentionally does not include robot-specific profiles, task alignment helpers, safety harnesses, standby policies, or vendor SDK plugins. Those features can be built as downstream packages on top of the core interfaces.

## Minimal Usage

```python
from metasim.protocol_sim.core import (
    CommandAdapter,
    MetaSimProtocolAdapter,
    ProtocolCodec,
    ProtocolServer,
    ProtocolServerConfig,
    Transport,
)

server = ProtocolServer(
    sim=MetaSimProtocolAdapter(handler),
    transport=my_transport,
    codec=my_codec,
    command_adapter=my_command_adapter,
    config=ProtocolServerConfig(dt=0.01, realtime=True),
)

server.start()
try:
    server.run_forever()
finally:
    server.close()
```
