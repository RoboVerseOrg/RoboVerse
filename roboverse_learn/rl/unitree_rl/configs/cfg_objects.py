from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveSphereCfg


class BallCfg(PrimitiveSphereCfg):
    def __init__(self):
        super().__init__(
            name="ball",
            radius=0.05,
            color=[1.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        )
        self.mass = 1.0
        self.enabled_gravity = True
        self.collision_enabled = True

        self.init_position = [2.0, 0.0, 1.0]
        self.init_rotation = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        self.init_velocity = [-10.0, 0.0, 0.0]  # vx, vy, vz
        self.angular_velocity = [0.0, 0.0, 0.0]  # wx, wy, wz

        self.root_state = [*self.init_position, *self.init_rotation, *self.init_velocity, *self.angular_velocity]  # pos(3), rot(4), vel(3), ang_vel(3)
