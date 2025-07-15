import os

import numpy as np
from dm_control.rl import control
from dm_control.suite import base, common, fish
from dm_control.utils import io as resources
from dm_control.utils import rewards

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")

_DEFAULT_TIME_LIMIT = 40
_CONTROL_TIMESTEP = 0.04
_JOINTS = [
    "tail1",
    "tail_twist",
    "tail2",
    "finright_roll",
    "finright_pitch",
    "finleft_roll",
    "finleft_pitch",
]


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, "fish.xml")), common.ASSETS


@fish.SUITE.add("custom")
def obstacles(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Fish Obstacles task."""
    physics = fish.Physics.from_xml_string(*get_model_and_assets())
    task = Obstacles(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit, **environment_kwargs
    )


class Obstacles(fish.Swim):
    """A custom Fish Obstacles task."""

    def __init__(self, random=None):
        super().__init__(random=random)

    def in_wall(self, physics, name, min_distance=0.08):
        """Returns True if the given body is too close to a wall."""
        for wall in ["wall0", "wall1", "wall2", "wall3"]:
            l1_dist = np.min(np.abs(physics.named.data.geom_xpos[name][:2] - physics.named.data.geom_xpos[wall][:2]))
            if l1_dist < min_distance:
                return True
        return False

    def initialize_episode(self, physics):
        in_wall = True
        while in_wall:
            # Randomize fish position.
            quat = self.random.randn(4)
            physics.named.data.qpos["root"][3:7] = quat / np.linalg.norm(quat)
            for joint in _JOINTS:
                physics.named.data.qpos[joint] = self.random.uniform(-0.2, 0.2)
            # Randomize target position.
            physics.named.model.geom_pos["target", "x"] = self.random.uniform(-0.4, 0.4)
            physics.named.model.geom_pos["target", "y"] = self.random.uniform(-0.4, 0.4)
            physics.named.model.geom_pos["target", "z"] = self.random.uniform(0.1, 0.3)
            # Make sure target is not too close to a wall.
            physics.after_reset()
            in_wall = self.in_wall(physics, "target")
        base.Task.initialize_episode(self, physics)

    def get_reward(self, physics):
        radii = physics.named.model.geom_size[["mouth", "target"], 0].sum()
        in_target = rewards.tolerance(
            np.linalg.norm(physics.mouth_to_target()),
            bounds=(0, radii),
            margin=2 * radii,
        )
        is_upright = 0.5 * (physics.upright() + 1)
        is_not_in_wall = 1.0 - self.in_wall(physics, "torso", min_distance=0.06)
        return is_not_in_wall * (7 * in_target + is_upright) / 8
