"""Regression: ``ManagerBasedRVEnv`` is reachable through the gym API.

Two defects made every manager-based (mjlab v2) task unusable via ``gym.make``:

* ``ManagerBasedRVEnv.reset`` set ``self._obs_buf`` and fell off the end, returning
  ``None``. ``metasim.task.gym_registration`` does ``obs, info = task_env.reset(seed=seed)``,
  which raised ``TypeError: cannot unpack non-iterable NoneType object``. The base
  ``RLTaskEnv.reset`` returns ``(obs, info)``; the manager base must too.
* ``ManagerBasedRVEnv`` bypasses ``RLTaskEnv.__init__`` (managers must be wired before the
  first ``_observation``), so ``num_obs`` was never set and the inherited
  ``observation_space`` raised ``AttributeError: 'num_obs'`` inside ``GymEnvWrapper.__init__``
  — ``gym.make`` failed before a task was ever reset.

Also guards the ``reset(seed=)`` contract end-to-end: the seed must reach the reset noise,
not merely be forwarded into a handler whose RNG nobody draws from.

Uses the cartpole scene-MJCF task on the MuJoCo backend (lightest manager-based task);
skips without ``mujoco``.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mujoco")

TASK = "mjlab.cartpole_balance_v2"


@pytest.fixture(scope="module")
def gym_env():
    import gymnasium as gym

    import roboverse_pack.tasks.mjlab  # noqa: F401  (registers the mjlab tasks)
    from metasim.task.gym_registration import register_task_with_gym

    env = gym.make(register_task_with_gym(TASK), simulator="mujoco", headless=True)
    yield env
    env.close()


def test_gym_reset_returns_obs_and_info(gym_env):
    """``gym.make(...).reset(seed=)`` unpacks — the TypeError regression."""
    obs, info = gym_env.reset(seed=0)

    assert isinstance(obs, dict) and obs, "reset must return the per-group observation dict"
    for group in ("actor", "critic"):
        assert isinstance(obs[group], torch.Tensor), f"obs['{group}'] must be a tensor"
        assert obs[group].shape[0] == gym_env.unwrapped.task_env.num_envs
    assert isinstance(info, dict)
    assert "observations" in info, "info must carry the rsl-rl observation entry"


def test_reset_obs_has_same_structure_as_step_obs(gym_env):
    """A gym caller must see the same observation type from reset and step."""
    reset_obs, _ = gym_env.reset(seed=0)
    num_actions = gym_env.unwrapped.task_env.num_actions
    step_obs, _, _, _, _ = gym_env.step(torch.zeros(1, num_actions))

    assert sorted(reset_obs) == sorted(step_obs)
    for group, tensor in reset_obs.items():
        assert tensor.shape == step_obs[group].shape


def test_observation_space_describes_the_returned_obs(gym_env):
    """``observation_space`` must exist (gym.make reads it) and match the obs groups."""
    from gymnasium import spaces

    space = gym_env.observation_space
    assert isinstance(space, spaces.Dict)

    obs, _ = gym_env.reset(seed=0)
    assert sorted(space.spaces) == sorted(obs)
    for group, tensor in obs.items():
        assert space[group].shape == (tensor.shape[-1],)


def test_reset_seed_is_reproducible(gym_env):
    """Equal seeds → identical reset state; different seeds → different reset state."""
    first, _ = gym_env.reset(seed=7)
    a = first["actor"].clone()
    second, _ = gym_env.reset(seed=7)
    assert torch.allclose(a, second["actor"]), "reset(seed=N) must reproduce the same initial state"

    other, _ = gym_env.reset(seed=8)
    assert not torch.allclose(a, other["actor"]), "a different seed must give a different reset state"


def test_a_raising_sensor_or_command_manager_surfaces_from_step_and_reset(gym_env):
    """The manager base used to wrap every manager / sensor update and reset in ``except Exception:
    pass``, so a broken sensor fed zeroed contact history into the reward with no sign of it."""

    class _Broken:
        def update(self, dt):
            raise RuntimeError("sensor broke")

        def reset(self, env_ids):
            raise RuntimeError("sensor broke")

    task_env = gym_env.unwrapped.task_env
    gym_env.reset(seed=0)
    task_env._mjlab_sensors["_broken"] = _Broken()
    try:
        with pytest.raises(RuntimeError, match="sensor broke"):
            gym_env.step(torch.zeros(1, gym_env.unwrapped.task_env.num_actions))
        with pytest.raises(RuntimeError, match="sensor broke"):
            gym_env.reset(seed=0)
    finally:
        del task_env._mjlab_sensors["_broken"]
    gym_env.reset(seed=0)
