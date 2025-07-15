import os

import numpy as np
from dm_control.rl import control
from dm_control.suite import common, hopper
from dm_control.utils import io as resources
from dm_control.utils import rewards

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")

_CONTROL_TIMESTEP = 0.02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2

# Angular momentum above which reward is 1.
_SPIN_SPEED = 5


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, "hopper.xml")), common.ASSETS


@hopper.SUITE.add("custom")
def hop_backwards(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Hop Backwards task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopper(goal="hop-backwards", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@hopper.SUITE.add("custom")
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopper(goal="flip", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@hopper.SUITE.add("custom")
def flip_backwards(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip Backwards task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopper(goal="flip-backwards", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


class Physics(hopper.Physics):
    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom["torso"][1]


class CustomHopper(hopper.Hopper):
    """Custom Hopper tasks."""

    def __init__(self, goal="hop-backwards", random=None):
        super().__init__(None, random)
        self._goal = goal

    def _hop_backwards_reward(self, physics):
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        hopping = rewards.tolerance(
            physics.speed(),
            bounds=(-float("inf"), -_HOP_SPEED / 2),
            margin=_HOP_SPEED / 4,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return standing * hopping

    def _flip_reward(self, physics, forward=True):
        reward = rewards.tolerance(
            (1.0 if forward else -1.0) * physics.angmomentum(),
            bounds=(_SPIN_SPEED, float("inf")),
            margin=_SPIN_SPEED / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        return reward

    def get_reward(self, physics):
        if self._goal == "hop-backwards":
            return self._hop_backwards_reward(physics)
        elif self._goal == "flip":
            return self._flip_reward(physics, forward=True)
        elif self._goal == "flip-backwards":
            return self._flip_reward(physics, forward=False)
        else:
            raise NotImplementedError(f"Goal {self._goal} is not implemented.")


if __name__ == "__main__":
    env = hop_backwards()
    obs = env.reset()
    import numpy as np

    next_obs, reward, done, info = env.step(np.zeros(2))
    print(reward)
