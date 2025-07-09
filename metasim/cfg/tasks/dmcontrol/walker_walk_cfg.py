import torch

from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class WalkerWalkCfg(BaseTaskCfg):
    episode_length = 300
    objects = []
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    # Constants from DeepMind Control Suite
    _STAND_HEIGHT = 1.2  # Target height for standing reward
    _MOVE_SPEED = 1.0  # Target horizontal speed
    _TERMINATION_HEIGHT = 0.8  # Height at which the walker is considered to have fallen

    def _sigmoid(self, x, value_at_margin=0.1, sigmoid_type="gaussian"):
        """Implements sigmoid functions for the tolerance reward calculation."""
        if sigmoid_type == "gaussian":
            scale = -torch.log(torch.tensor(value_at_margin))
            return torch.exp(-(x**2) * scale)
        elif sigmoid_type == "linear":
            scale = 1.0 - value_at_margin
            return torch.clip(1.0 - x * scale, 0.0, 1.0)
        else:
            # Default to gaussian if unrecognized
            scale = -torch.log(torch.tensor(value_at_margin))
            return torch.exp(-(x**2) * scale)

    def _tolerance(self, x, bounds, margin, sigmoid="gaussian", value_at_margin=0.1):
        """PyTorch version of tolerance function from DM Control."""
        lower, upper = bounds

        # Check if x is within bounds
        in_bounds = torch.logical_and(lower <= x, x <= upper)

        if margin == 0:
            value = torch.where(in_bounds, torch.tensor(1.0), torch.tensor(0.0))
        else:
            # Distance to nearest bound, normalized by margin
            d = torch.where(
                x < lower, (lower - x) / margin, torch.where(x > upper, (x - upper) / margin, torch.tensor(0.0))
            )
            # Apply sigmoid to the distance
            value = torch.where(in_bounds, torch.tensor(1.0), self._sigmoid(d, value_at_margin, sigmoid))

        return value

    def reward_fn(self, states):
        rewards = []
        for env_state in states:
            # Get walker states
            walker_state = env_state["walker"]

            # Extract positions, rotations and velocities
            pos = walker_state["pos"]
            rot = walker_state["rot"]  # quaternion [w, x, y, z]
            vel = walker_state["vel"]

            # 1. Calculate torso height (z position)
            torso_height = pos[2]

            # 2. Calculate upright reward (torso z-axis alignment with world z-axis)
            # Convert quaternion to get torso's up vector
            if len(rot) == 4:  # quaternion [w, x, y, z]
                w, x, y, z = rot
                # z-component of the rotated unit z vector
                torso_upright = 1.0 - 2.0 * (x**2 + y**2)
            else:  # rotation matrix or other format
                # Assuming the third row of rotation matrix represents up vector
                torso_upright = rot[2]

            # 3. Standing reward
            standing = self._tolerance(
                torso_height, bounds=(self._STAND_HEIGHT, float("inf")), margin=self._STAND_HEIGHT / 2, sigmoid="linear"
            )

            # 4. Normalize upright measure to [0, 1]
            upright = (1.0 + torso_upright) / 2.0

            # 5. Combine standing and upright for the stand reward
            stand_reward = (3.0 * standing + upright) / 4.0

            # 6. Calculate horizontal velocity (speed in x direction)
            horizontal_velocity = vel[0]  # X velocity

            # 7. Velocity reward using tolerance function
            move_reward = self._tolerance(
                horizontal_velocity,
                bounds=(self._MOVE_SPEED, float("inf")),
                margin=self._MOVE_SPEED / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )

            # 8. Combine stand and move rewards
            total_reward = stand_reward * (5.0 * move_reward + 1.0) / 6.0

            # 9. Check for termination height
            termination_height = 0.8  # If the walker falls below this height
            if torso_height < termination_height:
                total_reward = torch.tensor(0.0)  # Zero reward for falling

            rewards.append(total_reward)

        # Return tensor of rewards
        return torch.stack(rewards)

    # def termination_fn(self, states):
    #     terminations = []
    #     for env_state in states:
    #         # Get walker states
    #         walker_state = env_state["walker"]
    #         torso_height = walker_state["pos"][2]
    #         termination = torso_height < self._TERMINATION_HEIGHT
    #         terminations.append(termination)
    #     return torch.stack(terminations)
