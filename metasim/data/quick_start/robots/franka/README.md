The `franka_v1.usd` is the first version of franka used in both isaaclab version and metasim version. It comes from the Isaac Sim official model. Its old name in data_isaaclab is franka_instanceable_flattened.usd, refered by franka_cfg.py. It works well only with `franka_isaaclab_cfg.py`, which is the config for isaaclab.

The `franka_v2.usd` is the second version of franka used in metasim version. It comes from the isaacsim version of this repo. The damping and stiffness of both the arm and the gripper are modified. Its old name in data_isaaclab is franka_new.usd, refered by franka_stable_cfg.py.

`franka_v2.usd` is tuned to be more stable, and is expected to have better performance for tasks that is very delicate. It is tested on `square_d0` task from RoboSuite. The `franka_v1.usd` has ~10% success rate, while `franka_v2.usd` has ~25% success rate.
