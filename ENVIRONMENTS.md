# ENVIRONMENTS.md

This file defines the local environment correspondence for simulator-backed development and testing.

Update it when the actual conda env names on the machine differ from the defaults below.

## Simulator Environment Correspondence

- `isaacgym` -> conda env `isaacgym`
- `mujoco` -> conda env `isaacgym`
- `newton` -> conda env `newton`
- `isaacsim` -> conda env `isaacsim`

## Notes

- Keep this file simple and human-editable.
- If a simulator env is unknown or not configured, ask the user before running simulator-backed tests.
