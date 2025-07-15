import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import ball_in_cup, base, common
from dm_control.utils import io as resources
from dm_control.utils import rewards

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")

_DIST_TARGET = 0.5
_TARGET_SPEED = 6.0

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = 0.02  # (seconds)


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return (
        resources.GetResource(os.path.join(_TASKS_DIR, "ball_in_cup.xml")),
        common.ASSETS,
    )


@ball_in_cup.SUITE.add("custom")
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Ball-in-Cup Spin task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomBallInCup(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics with additional features for the Ball-in-Cup domain."""

    def ball_to_target(self):
        """Returns the vector from the ball to the target."""
        target = self.named.data.site_xpos["target", ["x", "z"]]
        ball = self.named.data.xpos["ball", ["x", "z"]]
        return target - ball

    def in_target(self):
        """Returns 1 if the ball is in the target, 0 otherwise."""
        ball_to_target = abs(self.ball_to_target())
        target_size = self.named.model.site_size["target", [0, 2]]
        ball_size = self.named.model.geom_size["ball", 0]
        return float(all(ball_to_target < target_size - ball_size))


class CustomBallInCup(ball_in_cup.BallInCup):
    """Custom Ball-in-Cup tasks."""

    def initialize_episode(self, physics):
        # Find a collision-free random initial position of the ball.
        penetrating = True
        valid_pos = False
        init_out_of_target = self.random.uniform() < 0.1
        while penetrating or not valid_pos:
            # Assign a random ball position.
            physics.named.data.qpos["ball_x"] = self.random.uniform(-0.2, 0.2)
            physics.named.data.qpos["ball_z"] = self.random.uniform(0.2, 0.5)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
            valid_pos = bool(physics.in_target()) or init_out_of_target
        base.Task.initialize_episode(self, physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        dist = np.linalg.norm(physics.ball_to_target())
        ball_vel_x = abs(physics.named.data.qvel["ball_x"])
        ball_vel_z = abs(physics.named.data.qvel["ball_z"])
        ball_vel = np.linalg.norm([ball_vel_x, ball_vel_z])

        # reward: spin around target (maximize distance to target + ball velocity)
        dist_reward = rewards.tolerance(
            dist,
            bounds=(_DIST_TARGET, float("inf")),
            margin=_DIST_TARGET / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        not_in_target = 1 - physics.in_target()
        vel_reward = rewards.tolerance(
            ball_vel,
            bounds=(_TARGET_SPEED, float("inf")),
            margin=_TARGET_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        spin_reward = not_in_target * (dist_reward + 2 * vel_reward) / 3
        return spin_reward
