import os

import numpy as np
from dm_control.rl import control
from dm_control.suite import common, pendulum
from dm_control.utils import io as resources
from dm_control.utils import rewards

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")

_DEFAULT_TIME_LIMIT = 20
_TARGET_SPEED = 9.0


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return (
        resources.GetResource(os.path.join(_TASKS_DIR, "pendulum.xml")),
        common.ASSETS,
    )


@pendulum.SUITE.add("custom")
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns pendulum spin task."""
    physics = pendulum.Physics.from_xml_string(*get_model_and_assets())
    task = Spin(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Spin(pendulum.SwingUp):
    """A custom Pendulum Spin task."""

    def __init__(self, random=None):
        super().__init__(random=random)

    def get_reward(self, physics):
        return rewards.tolerance(
            np.linalg.norm(physics.angular_velocity()),
            bounds=(_TARGET_SPEED, float("inf")),
            margin=_TARGET_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
