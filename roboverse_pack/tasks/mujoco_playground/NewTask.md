## 0.  Materials You Have

1. **Original task source code** (MuJoCoPlayground version).
2. **Minimal example of the target framework** (RoboVerse version).
3. **RoboVerse state documentation** (schema description).
4. **Common API → schema mapping table**.

------

### 1. Quick Scan — Source Code Audit

- **Locate four logic blocks**
  - reset / initial state
  - observation construction
  - reward & termination
  - step / control details
- **Enumerate all low-level dependencies**
  - qpos/qvel/ctrl indices
  - site / geom / body queries
  - contact / sensor flags
  - keyframe / XML references

------

### 2. Filter & Categorize — Decide Alignment Strategy

| Category                   | Handling Strategy                                            |
| -------------------------- | ------------------------------------------------------------ |
| **Directly mappable**      | Analyze the physical semantics and mapping relations on both sides (e.g., qpos → `joint_pos`). |
| **Requires query**         | Register the corresponding Site/Contact queries in `_extra_spec()`. |
| **Needs refactor**         | For example, keyframe initial pose → manually fill in `_get_initial_states()`. |
| **Missing / cannot align** | Mark as TODO, leave for human or later iteration.            |

------

### 3. Scenario Setup — Scenario & Asset

1. **Asset localization**: Identify meshes / URDF used in the original XML, map them to RoboVerse’s `Primitive*Cfg` or external URDF path.
2. **Scenario scaffold**: Generate `ScenarioCfg` skeleton:
   - `objects`, `robots` lists
   - `sim_params` (dt, etc.)
   - `decimation` / `max_episode_steps`
3. **Random initialization logic**: If the original task has random reset, implement ranges in `_prepare_states()`.

------

### 4. State Adapter — Add Observations / Queries

- **Add extras**: For anything not covered by the RoboVerse state (e.g., site/geom pose or contact flag), register it in `_extra_spec()` with `SitePos|Mat|ContactData`, so it’s accessible in reward/obs.
- **Validate schema**: Ensure all reward/obs fields are accessible via the RoboVerse state. Missing fields must either be registered as extras or implemented with a custom query.

------

### 5. Logic Translation — Reward / Observation / Termination

1. **Line-by-line replacement**: According to the translation plan, replace all raw simulator API calls with schema access.
2. **Maintain equivalence**: Do not invent new shaping; unless the original reward clearly depends on non-replicable sim details, in which case leave a comment explaining the adjustment.
3. **Annotate original source**: For each replacement, leave one line `# mj: ...` for traceability.
4. **Mark gaps**: If translation is impossible or unclear, leave explicit TODOs.
5. **Extract utils**: For code patterns used repeatedly, factor them into utility functions, but keep the overall structure consistent with the source code

# Appendix

**1.Original Task Code**

```python
def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
          )
      ),
      impl='jax',
      nconmax=24 * 8192,
      njmax=128,
  )
  return config

class PandaPickCube(panda.PandaBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.0]),
            maxval=jp.array([0.2, 0.2, 0.0]),
        )
        + self._init_obj_pos
    )

    # initialize target position
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.2]),
            maxval=jp.array([0.2, 0.2, 0.4]),
        )
        + self._init_obj_pos
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs
```



**2.Target Code**

```python
DEFAULT_CONFIG = {
    "action_scale": 0.04,
    "reward_config": {
        "scales": {
            # Gripper goes to the box and closes when close enough.
            "gripper_box": 4.0,
            # Box goes to the target mocap.
            "box_target": 8.0,
            # Do not collide the gripper with the floor.
            "no_floor_collision": 0.25,
            # Arm stays close to target pose.
            "robot_target_qpos": 0.3,
        }
    },
    "nconmax": 24 * 8192,
    "njmax": 128,
}

@register_task("mujoco_playground.pick", "pick_rl")
class PandaPickCube(RLTaskEnv):
    """Bring a box to a target. """

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="box",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveSphereCfg(
                name="target",
                mass=0.01,
                radius=0.02,
                physics=PhysicStateType.XFORM,
                color=(0.7, 1.0, 0.7),
            ),
        ],
        robots=["franka"],
        sim_params=SimParamCfg(
            dt=0.005,
            nconmax=24 * 8192,
            njmax=128,
        ),
        decimation=4,
    )
    max_episode_steps = 150

    def __init__(self, scenario, device=None):
        self.robot_name = self.scenario.robots[0].name
        self._last_action = None  # Initialize _last_action to None
        super().__init__(scenario, device)
        self._action_scale = DEFAULT_CONFIG.get("action_scale", 0.04)
        self.reward_functions = [
            self._reward_box_target,
            self._reward_gripper_box,
            self._reward_no_floor_collision,
            self._reward_robot_target_qpos,
        ]
        self.reward_weights = [
            DEFAULT_CONFIG["reward_config"]["scales"]["box_target"],
            DEFAULT_CONFIG["reward_config"]["scales"]["gripper_box"],
            DEFAULT_CONFIG["reward_config"]["scales"]["no_floor_collision"],
            DEFAULT_CONFIG["reward_config"]["scales"]["robot_target_qpos"],
        ]

    def _prepare_states(self, states, env_ids):
        """Preprocess initial states, randomizing positions within specified ranges."""

        box_pos_range = torch.tensor([[0.4, -0.2, 0.0], [0.8, 0.2, 0.0]], device=self.device)
        target_pos_range = torch.tensor([[0.4, -0.2, 0.2], [0.8, 0.2, 0.4]], device=self.device)

        box_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (box_pos_range[1] - box_pos_range[0]) + box_pos_range[0]
        )
        target_pos = (
            torch.rand(self.num_envs, 3, device=self.device) * (target_pos_range[1] - target_pos_range[0])
            + target_pos_range[0]
        )

        box_quat = states.objects["box"].root_state[:, 3:7]
        target_quat = states.objects["target"].root_state[:, 3:7]
        zero_vel = torch.zeros(self.num_envs, 3, device=self.device)
        zero_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        states.objects["box"].root_state = torch.cat([box_pos, box_quat, zero_vel, zero_ang_vel], dim=-1)
        states.objects["target"].root_state = torch.cat([target_pos, target_quat, zero_vel, zero_ang_vel], dim=-1)

        return states

    def reset(self, env_ids=None):
        """Reset environment and last actions."""
        if env_ids is None or self._last_action is None:
            self._last_action = self._initial_states.robots[self.robot_name].joint_pos[:, :]
        else:
            self._last_action[env_ids] = self._initial_states.robots[self.robot_name].joint_pos[env_ids, :]
        return super().reset(env_ids=env_ids)

    def step(self, actions):
        """Step with delta control."""
        delta_actions = actions * self._action_scale
        new_actions = self._last_action + delta_actions
        real_actions = torch.maximum(torch.minimum(new_actions, self._action_high), self._action_low)
        obs, reward, terminated, time_out, info = super().step(real_actions)
        self._last_action = real_actions
        return obs, reward, terminated, time_out, info

    def _reward_box_target(self, env_states) -> torch.Tensor:
        """Reward for bringing box to target position and orientation."""
        # Get states from tensor state
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["box"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        target_quat = env_states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

        # Get finger joint positions (first two joints are finger joints)

        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)
        pos_err = torch.norm(target_pos - box_pos, dim=-1)  # [num_envs]
        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        target_mat = matrix_from_quat(target_quat)  # [num_envs, 3, 3]

        rot_err = torch.norm(
            target_mat.view(self.num_envs, -1)[:, :6] - box_mat.view(self.num_envs, -1)[:, :6], dim=-1
        )  # [num_envs]
        box_target = 1 - torch.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        reached_box = (torch.norm(box_pos - gripper_pos, dim=-1) < 0.012).float()

        return box_target * reached_box

    def _reward_gripper_box(self, env_states) -> torch.Tensor:
        """Reward for gripper approaching the box and closing when close enough."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        gripper_pos = env_states.extras["gripper_pos"]  # [num_envs, 3]
        gripper_box_dist = torch.norm(box_pos - gripper_pos, dim=-1)  # [num_envs]
        approach_reward = 1 - torch.tanh(5 * gripper_box_dist)
        return approach_reward 

    def _reward_no_floor_collision(self, env_states) -> torch.Tensor:
        """Reward for not colliding with the floor."""
        # Get individual floor collision sensor data from extra observations
        extra_data = env_states.extras
        left_collision = extra_data["left_finger_floor_collision"]  # [num_envs]
        right_collision = extra_data["right_finger_floor_collision"]  # [num_envs]
        hand_collision = extra_data["hand_capsule_floor_collision"]  # [num_envs]
        any_collision = (left_collision > 0) | (right_collision > 0) | (hand_collision > 0)  # [num_envs]
        no_floor_collision = (~any_collision).float()
        return no_floor_collision

    def _reward_robot_target_qpos(self, env_states) -> torch.Tensor:
        """Reward for robot staying close to target joint positions."""
        # [num_envs, ARM_joints],first two are finger joints
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos[:, 2:]
        target_joint_pos = self._initial_states.robots[self.robot_name].joint_pos[:, 2:]

        joint_error = torch.norm(robot_joint_pos - target_joint_pos, dim=-1)  # [num_envs]
        return 1 - torch.tanh(joint_error)

    def _observation(self, env_states) -> torch.Tensor:
        """Get observation using RoboVerse tensor state."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        box_quat = env_states.objects["box"].root_state[:, 3:7]  # [num_envs, 4]
        target_pos = env_states.objects["target"].root_state[:, 0:3]  # [num_envs, 3]
        target_quat = env_states.objects["target"].root_state[:, 3:7]  # [num_envs, 4]

        gripper_pos = env_states.extras["gripper_pos"]  # (B, 3)
        gripper_mat = env_states.extras["gripper_mat"]  # (B, 9)
        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]

        # Convert quaternions to rotation matrices for box and target
        from metasim.utils.math import matrix_from_quat

        box_mat = matrix_from_quat(box_quat)  # [num_envs, 3, 3]
        target_mat = matrix_from_quat(target_quat)  # [num_envs, 3, 3]

        box_mat_flat = box_mat.view(self.num_envs, -1)  # [num_envs, 9]
        target_mat_flat = target_mat.view(self.num_envs, -1)  # [num_envs, 9]

]        # Ensure gripper_mat has correct shape [num_envs, 9
        if gripper_mat.dim() == 3:
            gripper_mat = gripper_mat.view(self.num_envs, -1)  # Reshape to [num_envs, 9]

        box_to_gripper = box_pos - gripper_pos  # [num_envs, 3]
        target_to_box = target_pos - box_pos  # [num_envs, 3]
        target_box_rot_diff = target_mat_flat[:, :6] - box_mat_flat[:, :6]  # [num_envs, 6]

        # Get joint position targets and calculate control error (target - actual)
        robot_joint_pos_target = env_states.robots[self.robot_name].joint_pos_target  # [num_envs, num_joints]
        joint_pos_error = robot_joint_pos_target - robot_joint_pos  # [num_envs, num_joints]

        obs = torch.cat(
            [
                robot_joint_pos,  # Joint positions [num_envs, num_joints]
                robot_joint_vel,  # Joint velocities [num_envs, num_joints]
                gripper_pos,  # Robot position [num_envs, 3]
                gripper_mat[:, 3:],  # Robot orientation (last 6 elements) [num_envs, 6]
                box_mat_flat[:, 3:],  # Box orientation (last 6 elements) [num_envs, 6]
                box_to_gripper,  # Box to gripper vector [num_envs, 3]
                target_to_box,  # Target to box vector [num_envs, 3]
                target_box_rot_diff,  # Target-box orientation difference [num_envs, 6]
                joint_pos_error,  # Joint position error (target - actual) [num_envs, num_joints]
            ],
            dim=-1,
        )  # [num_envs, obs_dim]

        return obs

    def _terminated(self, env_states) -> torch.Tensor:
        """Check if episode should be terminated."""
        box_pos = env_states.objects["box"].root_state[:, 0:3]  # [num_envs, 3]
        out_of_bounds = torch.any(torch.abs(box_pos) > 1.0, dim=-1)  # [num_envs]
        out_of_bounds |= box_pos[:, 2] < 0.0  # Box below ground

        robot_joint_pos = env_states.robots[self.robot_name].joint_pos  # [num_envs, num_joints]
        robot_joint_vel = env_states.robots[self.robot_name].joint_vel  # [num_envs, num_joints]
        nan_joints = torch.isnan(robot_joint_pos).any(dim=-1) | torch.isnan(robot_joint_vel).any(dim=-1)

        nan_objects = torch.isnan(box_pos).any(dim=-1)
        terminated = out_of_bounds | nan_joints | nan_objects

        return terminated

    def _get_initial_states(self) -> list[dict] | None:
        """Get initial states for all environments."""
        init = [
            {
                "objects": {
                    "box": {
                        "pos": torch.tensor([0.5, 0.0, 0.03]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "target": {
                        "pos": torch.tensor([0.5, 0.0, 0.03]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": 0.3,
                            "panda_joint3": 0.0,
                            "panda_joint4": -1.57079,
                            "panda_joint5": 0.0,
                            "panda_joint6": 2.0,
                            "panda_joint7": -0.7853,
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                        },
                    }
                },
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]

        return init

    def _extra_spec(self) -> dict:
        return {
            "left_finger_floor_collision": ContactData(f"{self.robot_name}/left_finger_pad", "floor"),
            "right_finger_floor_collision": ContactData(f"{self.robot_name}/right_finger_pad", "floor"),
            "hand_capsule_floor_collision": ContactData(f"{self.robot_name}/hand_capsule", "floor"),
            # Gripper site queries for position and orientation
            "gripper_pos": SitePos("gripper"),
            "gripper_mat": SiteMat("gripper"),
        }

    def _get_ee_state(self, states):
        """Return EE state using site queries."""
        ee_pos_world = states.extras["gripper_pos"]  # (B, 3)
        ee_mat_world = states.extras["gripper_mat"]  # (B, 9)
        return ee_pos_world, ee_mat_world
```

**3.RoboVerse state doc

### **Tensor State**

`TensorStates` is a dict of vectorized states. At the top level, it contains 4 fields: objects, robots, cameras and extras. Each field then contains multiple sub-fields for data.

`@dataclass class ObjectState: """State of a single object."""

```
root_state: torch.Tensor
"""Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
body_names: list[str] | None = None
"""Body names. This is only available for articulation objects."""
body_state: torch.Tensor | None = None
"""Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
joint_pos: torch.Tensor | None = None
"""Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
joint_vel: torch.Tensor | None = None
"""Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""
```

@dataclass class RobotState: """State of a single robot."""

```
root_state: torch.Tensor
"""Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
body_names: list[str]
"""Body names."""
body_state: torch.Tensor
"""Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
joint_pos: torch.Tensor
"""Joint positions. Shape is (num_envs, num_joints)."""
joint_vel: torch.Tensor
"""Joint velocities. Shape is (num_envs, num_joints)."""
joint_pos_target: torch.Tensor
"""Joint positions target. Shape is (num_envs, num_joints)."""
joint_vel_target: torch.Tensor
"""Joint velocities target. Shape is (num_envs, num_joints)."""
joint_effort_target: torch.Tensor
"""Joint effort targets. Shape is (num_envs, num_joints)."""
```

@dataclass class CameraState:

@dataclass class TensorState: objects: dict[str, ObjectState] robots: dict[str, RobotState] cameras: dict[str, CameraState] extras: dict = field(default_factory=dict)`

**Dict State**

`DictState` is more user-friendly compared to `TensorState`, but scrifices efficiency.

`Dof = Dict[str, float]

class DictObjectState(TypedDict): """State of the object."""

```
pos: torch.Tensor
rot: torch.Tensor
vel: torch.Tensor | None
ang_vel: torch.Tensor | None
dof_pos: Dof | None
dof_vel: Dof | None
```

class DictRobotState(DictObjectState): """State of the robot."""

```
dof_pos: Dof | None
dof_vel: Dof | None

dof_pos_target: Dof | None
dof_vel_target: Dof | None
dof_torque: Dof | None
```

class DictEnvState(TypedDict): """State of the environment."""

```
objects: dict[str, DictObjectState]
robots: dict[str, DictRobotState]
cameras: dict[str, dict[str, torch.Tensor]]
extras: dict[str, Any]      # States of Extra information`
```

**4. Normal API → schema mapping**。

| MuJoCo `data` field             | RoboVerse state path                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `qpos`                          | `env_states.robots["<name>"].joint_pos` / `env_states.objects["<name>"].joint_pos` |
| `qvel`                          | `env_states.robots["<name>"].joint_vel` / `env_states.objects["<name>"].joint_vel` |
| `ctrl`                          | `env_states.robots["<name>"].joint_pos_target`               |
| `actuator_force`                | `env_states.robots["<name>"].joint_effort_target`            |
| `xpos`                          | `...root_state[..., 0:3]` ⇢ global position                  |
| `xquat`                         | `...root_state[..., 3:7]` ⇢ global orientation (w x y z)     |
| `cvel[..., 3:6]` (linear part)  | `...root_state[..., 7:10]` ⇢ linear vel (world)              |
| `cvel[..., 0:3]` (angular part) | `...root_state[..., 10:13]` ⇢ angular vel (world)            |