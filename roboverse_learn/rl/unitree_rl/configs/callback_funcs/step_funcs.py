from __future__ import annotations

from collections import deque
import torch

from metasim.types import TensorState
from metasim.utils.math import (
    quat_apply,
    wrap_to_pi,
    sample_uniform,
)

from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg


def resample_commands(env: EnvTypes, env_states: TensorState = None):
    """Randomly select commands for some environments.

    Args:
        env_ids (List[int]): Environments ids for which new commands are needed.
    """
    cfg: BaseEnvCfg.Commands = env.commands_manager
    if cfg.value is None:
        cfg.value = torch.zeros(
            size=(env.num_envs, cfg.num_commands), dtype=torch.float, device=env.device
        )
    ranges_tensor = torch.tensor(
        [
            cfg.ranges.lin_vel_x,
            cfg.ranges.lin_vel_y,
            cfg.ranges.heading if cfg.heading_command else cfg.ranges.ang_vel_yaw,
        ],
        device=env.device,
    )

    env_ids = (
        (env._episode_steps % int(cfg.resampling_time / env.step_dt) == 0)
        .nonzero(as_tuple=False)
        .flatten()
    )
    if len(env_ids) == 0:
        return

    cfg.value[env_ids, :] = sample_uniform(
        ranges_tensor[:, 0],
        ranges_tensor[:, 1],
        (len(env_ids), ranges_tensor.size(0)),
        device=env.device,
    )

    # low_cmd_mask = torch.norm(cfg.value[env_ids, :2], dim=1) < 0.1
    random_mask = (
        sample_uniform(0, 1, (len(env_ids),), device=env.device)
        <= cfg.rel_standing_envs
    )
    final_env_ids = random_mask.nonzero(as_tuple=False).flatten()
    cfg.value[env_ids][final_env_ids, :] = 0.0

    if cfg.heading_command:
        env_states = env.get_states() if env_states is None else env_states
        robot_state = env_states.robots[env.name]
        base_quat = robot_state.root_state[:, 3:7]
        forward = quat_apply(
            base_quat, env.forward_vec
        )  # quat:[w, x, y, z], forward:[x, y, z]
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cfg.value[:, 2] = torch.clip(
            0.5 * wrap_to_pi(cfg.value[:, 2] - heading), -1.0, 1.0
        )


def push_by_setting_velocity(
    env: EnvTypes,
    env_states: TensorState,
    interval_range_s: tuple | int = 5.0,
    velocity_range: list[list] = [[0] * 3, [0] * 3],
):
    """Randomly set robot's root velocity to simulate a push."""
    if not hasattr(env, "push_interval"):
        env.push_interval = (
            sample_uniform(
                interval_range_s[0],
                interval_range_s[1],
                (env.num_envs, 1),
                device=env.device,
            ).flatten()
            / env.step_dt
        ).to(torch.int)
    push_env_ids = (
        torch.logical_and(
            env._episode_steps % env.push_interval == 0, env._episode_steps > 0
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    if len(push_env_ids) == 0:
        return
    velocity_range = torch.tensor(velocity_range, device=env.device)
    # env_states = env.get_states()
    env_states.robots[env.name].root_state[push_env_ids, 7:10] += sample_uniform(
        velocity_range[0], velocity_range[1], (len(push_env_ids), 3), device=env.device
    )

    env.handler.set_states(env_states, push_env_ids.tolist())

class HistoryBuffer(deque):
    """A simple LIFO buffer that stores multiple tensors per entry under specified keys.

    Each pushed entry is a dict mapping each key -> tensor (cloned on push).

    Usage:
        buf = HistoryBuffer(maxlen=10, keys=("root_state", "obs"))
        buf.push({"root_state": t1, "obs": t2})
        last = buf.pop()  # returns the most-recent dict (LIFO)

    It also provides a convenience call interface to push data directly from
    env_states when used as a callback: `buf(env, env_states)` will read the
    attributes named in `keys` from `env_states.robots[env.name]` (or from the
    buffer instance itself if attributes with those names exist) and push them.
    """

    def __init__(self, maxlen: int | None, keys: tuple[str] | list[str] | str, name: str | None = None):
        # maxlen may be None for unbounded deque
        super().__init__(maxlen=maxlen)
        self.keys = tuple(keys) if isinstance(keys, (tuple, list)) else (keys,)
        # optional name used when reading from env_states: if provided, it will
        # be used as robot key; otherwise, the env.name value is used.
        self.name = name

    # Basic stack-style methods -------------------------------------------------
    def push(self, entry: dict):
        """Push an entry (dict of key->tensor). Clones tensor values on insert.

        Raises KeyError if any configured key is missing from the entry.
        """
        if not isinstance(entry, dict):
            raise TypeError("HistoryBuffer.push requires a dict of key->tensor")
        item = {}
        for k in self.keys:
            if k not in entry:
                raise KeyError(f"Missing key '{k}' for HistoryBuffer.push")
            v = entry[k]
            item[k] = v.clone() if isinstance(v, torch.Tensor) else v
        # append to the right; pop() will return the most-recent (LIFO)
        self.append(item)

    def pop_one(self) -> dict:
        """Pop and return the most-recently pushed entry (LIFO)."""
        return self.pop()

    def peek(self, index: int = 0) -> dict:
        """Peek at recently pushed entries without removing.

        index=0 -> last pushed, index=1 -> one-before-last, etc.
        """
        if index < 0 or index >= len(self):
            raise IndexError("peek index out of range")
        # -1 is last, -2 is one-before-last
        return self[-1 - index]

    # Convenience methods ------------------------------------------------------
    def push_from_env(self, env: EnvTypes, env_states: TensorState, env_name: str | None = None):
        """Collect configured keys from env_states.robots[env_name] (or from
        the buffer instance) and push them as one entry.
        """
        robot_name = env_name or self.name or getattr(env, "name", None)
        if robot_name is None:
            raise ValueError("No robot name available for HistoryBuffer.push_from_env")

        robot_state = env_states.robots[robot_name]
        entry = {}
        for k in self.keys:
            # priority: attribute on this buffer instance, else attribute on robot_state
            if hasattr(self, k):
                val = getattr(self, k)
            elif hasattr(robot_state, k):
                val = getattr(robot_state, k)
            else:
                raise ValueError(f"History buffer key {k} not found in task or robot states.")
            entry[k] = val.clone() if isinstance(val, torch.Tensor) else val

        self.push(entry)

    def __call__(self, env: EnvTypes, env_states: TensorState, *args, **kwds):
        """Callable convenience so this object can be used as a callback.

        It reads the configured keys from env_states.robots[env.name] (or from
        attributes on the instance) and pushes a dict entry.
        """
        self.push_from_env(env, env_states)

    # keep clear and len behavior from deque; add explicit alias
    def clear_all(self):
        super().clear()
