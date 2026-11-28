# Robots

Generated from `roboverse_pack/robots` and `results/robots/` by `generate_robot_docs.py`; do not edit by hand.
A robot is listed with the asset formats its config declares; `tools/assets/check_manifest.py` verifies on
every pull request that each declared file exists on the Hub. Robots whose assets did not exist were retired
on 2026-11-28 (`roboverse_pack/tasks/RETIRED_TASKS.md`). The self-check and parity tables come from the newest
result files per simulator; how to add a robot and pass them is in [Add a robot](../../start/add_a_robot.md).

## Configs

| config module | class | `name` | joints | asset formats |
|---|---|---|---|---|
| `aloha_agilex_cfg` | `AlohaAgilexCfg` | `aloha_agilex` | 16 |  |
| `arx_l5_mjcf_cfg` | `ArxL5Cfg` | `arx_l5` | 8 | MuJoCo, MJX |
| `fetch_cfg` | `FetchCfg` | `fetch` | 9 | URDF |
| `franka_cfg` | `FrankaCfg` | `franka` | 9 | MuJoCo, Isaac Sim, URDF, MJX |
| `franka_slide_urdf_cfg` | `FrankaSlideUrdfCfg` | `franka` | 9 | URDF |
| `franka_slide_urdf_sapien_cfg` | `FrankaSlideUrdfSapienCfg` | `franka` | 9 | URDF |
| `franka_urdf_cfg` | `FrankaUrdfCfg` | `franka` | 9 | URDF |
| `franka_urdf_sapien_cfg` | `FrankaUrdfSapienCfg` | `franka` | 9 | URDF |
| `franka_with_gripper_extension_cfg` | `FrankaWithGripperExtensionCfg` | `franka_with_gripper_extension` | 9 | Isaac Sim, URDF |
| `g1_cfg` | `G1Dof12Cfg` | `g1_dof12` | 12 | MuJoCo, Isaac Sim, URDF, MJX |
| `g1_cfg` | `G1Dof23Cfg` | `g1_dof23` | 23 |  |
| `g1_cfg` | `G1Dof27Cfg` | `g1_dof27` | 27 |  |
| `g1_cfg` | `G1Dof29Cfg` | `g1_dof29` | 29 | MuJoCo, Isaac Sim, URDF, MJX |
| `g1_cfg` | `G1Dof29Dex3Cfg` | `g1_dof29_dex3` | 43 |  |
| `g1_tracking` | `G1TrackingCfg` | `g1_tracking` | 29 | MuJoCo, Isaac Sim, URDF, MJX |
| `gen3_cfg` | `Gen3Cfg` | `gen3` | 7 | MuJoCo, Isaac Sim, URDF, MJX |
| `google_robot_static_cfg` | `GoogleRobotStaticCfg` | `google_robot_static` | 11 | URDF |
| `h1_body_collision` | `H1BodyCollisionCfg` | `h1` | 26 | MuJoCo, Isaac Sim, URDF, MJX |
| `h1_cfg` | `H1Cfg` | `h1` | 26 | MuJoCo, Isaac Sim, URDF, MJX |
| `h1_hand_cfg` | `H1HandCfg` | `h1_hand` | 76 | MuJoCo, MJX |
| `h1_simple_hand_cfg` | `H1SimpleHandCfg` | `h1_simple_hand` | 52 | MuJoCo, MJX |
| `h1_wrist_cfg` | `H1WristCfg` | `h1_wrist` | 19 | MuJoCo, Isaac Sim, URDF, MJX |
| `iiwa14_cfg` | `Iiwa14Cfg` | `iiwa14` | 7 | MuJoCo, Isaac Sim, URDF, MJX |
| `kinova_gen3_robotiq_2f85` | `KinovaGen3Robotiq2f85Cfg` | `kinova_gen3_robotiq_2f85` | 9 | Isaac Sim |
| `koch_mjcf_cfg` | `KochCfg` | `koch_1` | 6 | MuJoCo, MJX |
| `lite6_mjcf_cfg` | `Lite6Cfg` | `lite6` | 6 | MuJoCo, MJX |
| `mjlab_cartpole_cfg` | `MjlabCartpoleCfg` | `cartpole` | 2 | MuJoCo, MJX |
| `mjlab_g1_cfg` | `MjlabG1Cfg` | `g1` | 29 | MuJoCo, Isaac Sim, URDF, MJX |
| `mjlab_go1_cfg` | `MjlabGo1Cfg` | `go1` | 12 | MuJoCo, MJX |
| `mjlab_yam_cfg` | `MjlabYamCfg` | `yam` | 8 | MuJoCo, MJX |
| `openarm_wuji_cfg` | `OpenarmBimanualWujiCfg` | `openarm_bimanual_wuji` | 54 | MuJoCo, Isaac Sim, URDF, MJX |
| `openarm_wuji_cfg` | `OpenarmWujiCfg` | `openarm_wuji` | 54 | MuJoCo, Isaac Sim, URDF, MJX |
| `sawyer_cfg` | `SawyerCfg` | `sawyer` | 10 | Isaac Sim, URDF |
| `sawyer_mujoco_cfg` | `SawyerMujocoCfg` | `sawyer` | 9 | MuJoCo, Isaac Sim, MJX |
| `shadow_hand_cfg` | `ShadowHandCfg` | `shadow_hand` | 24 | MuJoCo, MJX |
| `sharpa_wave_cfg` | `SharpaWaveLeftCfg` | `sharpa_wave_left` | 22 | MuJoCo, MJX |
| `sharpa_wave_cfg` | `SharpaWaveRightCfg` | `sharpa_wave_right` | 22 | MuJoCo, MJX |
| `so_arm100_mjcf_cfg` | `SoArm100Cfg` | `so_arm100` | 6 | MuJoCo, MJX |
| `unitree_dex3_1_cfg` | `UnitreeDex31LeftCfg` | `unitree_dex3_1_left` | 7 | URDF |
| `ur10e_cfg` | `Ur10ECfg` | `ur10e` | 6 | MuJoCo, Isaac Sim, URDF, MJX |
| `ur5e_cfg` | `Ur5ECfg` | `ur5e` | 6 | MuJoCo, Isaac Sim, URDF, MJX |
| `xhand_cfg` | `XhandRightCfg` | `xhand_right` | 12 | URDF |
| `yam_mjcf_cfg` | `YamCfg` | `yam` | 8 | MuJoCo, MJX |
| `z1_cfg` | `Z1Cfg` | `z1` | 6 | MuJoCo, Isaac Sim, URDF, MJX |

44 robot configs.

## Self-check

### Self-check on isaacsim (2026-11-28)

`python -m metasim robot check --all --sim isaacsim`: 37 robots, 13 pass, 6 fail, 18 n/a (no asset declared for this backend, nothing else wrong). Source: `results/robots/2026-11-28-isaacsim-check/robots.json`.

| robot | config | assets | kinematics | limits | actuators | hold | verdict |
|---|---|---|---|---|---|---|---|
| `aloha_agilex` | FAIL | n/a | ok | ok | ok | n/a | FAIL |
| `arx_l5` | ok | n/a | ok | ok | ok | n/a | n/a |
| `cartpole` | ok | n/a | ok | ok | ok | n/a | n/a |
| `fetch` | ok | n/a | ok | ok | ok | n/a | n/a |
| `franka` | ok | ok | ok | ok | ok | ok | pass |
| `franka_with_gripper_extension` | ok | ok | ok | ok | ok | ok | pass |
| `g1` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof12` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof23` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_dof27` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_dof29` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof29_dex3` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_tracking` | FAIL | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `gen3` | ok | ok | ok | ok | ok | ok | pass |
| `go1` | ok | n/a | ok | ok | ok | n/a | n/a |
| `google_robot_static` | ok | n/a | ok | ok | ok | n/a | n/a |
| `h1` | ok | ok | ok | ok | ok | ok | pass |
| `h1_hand` | ok | n/a | ok | ok | ok | n/a | n/a |
| `h1_simple_hand` | ok | n/a | ok | ok | ok | n/a | n/a |
| `h1_wrist` | ok | ok | ok | ok | ok | ok | pass |
| `iiwa14` | ok | ok | ok | ok | ok | ok | pass |
| `kinova_gen3_robotiq_2f85` | ok | ok | ok | ok | ok | ok | pass |
| `koch_1` | ok | n/a | ok | ok | ok | n/a | n/a |
| `lite6` | ok | n/a | ok | ok | ok | n/a | n/a |
| `openarm_bimanual_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `openarm_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `sawyer` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `shadow_hand` | ok | n/a | ok | FAIL | FAIL | n/a | FAIL |
| `sharpa_wave_left` | ok | n/a | ok | ok | ok | n/a | n/a |
| `sharpa_wave_right` | ok | n/a | ok | ok | ok | n/a | n/a |
| `so_arm100` | ok | n/a | ok | ok | ok | n/a | n/a |
| `unitree_dex3_1_left` | ok | n/a | ok | ok | ok | n/a | n/a |
| `ur10e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e` | ok | ok | ok | ok | ok | ok | pass |
| `xhand_right` | ok | n/a | ok | ok | ok | n/a | n/a |
| `yam` | ok | n/a | ok | ok | ok | n/a | n/a |
| `z1` | ok | ok | ok | ok | ok | ok | pass |

Named failures (the first per robot and check):

- `aloha_agilex` / config: `no joint_limits for ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4', 'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint1', 'fr_joint2', 'fr_joint3', 'fr_joint4', 'fr_joint5', 'fr_joint6', 'fr_joint7', 'fr_`
- `g1_tracking` / config: `no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_shoulder_yaw_joint', 'left_wri`
- `g1_tracking` / assets: `['/workspace/roboverse_maint/RoboVerse/roboverse_data/unitree_description/usd/g1/g1.usd'] not local and not fetched from the Hub (Exception)`
- `g1_tracking` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'`
- `g1_tracking` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'`
- `g1_tracking` / hold: `not attempted: a cheaper check already failed`
- `openarm_bimanual_wuji` / assets: `['roboverse_data/robots/openarm_wuji/openarm_wuji.usd'] not local and not fetched from the Hub (Exception)`
- `openarm_bimanual_wuji` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_bimanual_wuji` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_bimanual_wuji` / hold: `not attempted: a cheaper check already failed`
- `openarm_wuji` / assets: `['roboverse_data/robots/openarm_wuji/openarm_wuji.usd'] not local and not fetched from the Hub (Exception)`
- `openarm_wuji` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_wuji` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_wuji` / hold: `not attempted: a cheaper check already failed`
- `sawyer` / hold: `child exited 1 without a report: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/sawyer/root_joint. Please check PhysX logs.`
- `shadow_hand` / limits: `MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 `
- `shadow_hand` / actuators: `MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 `

### Self-check on mujoco (2026-11-28)

`python -m metasim robot check --all --sim mujoco`: 37 robots, 19 pass, 8 fail, 10 n/a (no asset declared for this backend, nothing else wrong). Source: `results/robots/2026-11-28-mujoco-check/robots.json`.

| robot | config | assets | kinematics | limits | actuators | hold | verdict |
|---|---|---|---|---|---|---|---|
| `aloha_agilex` | FAIL | n/a | ok | ok | ok | n/a | FAIL |
| `arx_l5` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `cartpole` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `fetch` | ok | n/a | ok | ok | ok | n/a | n/a |
| `franka` | ok | ok | ok | ok | ok | ok | pass |
| `franka_with_gripper_extension` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof12` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof23` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_dof27` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_dof29` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof29_dex3` | ok | n/a | ok | ok | ok | n/a | n/a |
| `g1_tracking` | FAIL | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `gen3` | ok | ok | ok | ok | ok | ok | pass |
| `go1` | ok | ok | ok | ok | ok | ok | pass |
| `google_robot_static` | ok | n/a | ok | ok | ok | n/a | n/a |
| `h1` | ok | ok | ok | ok | ok | ok | pass |
| `h1_hand` | ok | ok | ok | ok | ok | ok | pass |
| `h1_simple_hand` | ok | ok | ok | ok | ok | ok | pass |
| `h1_wrist` | ok | ok | ok | ok | ok | ok | pass |
| `iiwa14` | ok | ok | ok | ok | ok | ok | pass |
| `kinova_gen3_robotiq_2f85` | ok | n/a | ok | ok | ok | n/a | n/a |
| `koch_1` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `lite6` | ok | ok | ok | ok | ok | ok | pass |
| `openarm_bimanual_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `openarm_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `sawyer` | ok | n/a | ok | ok | ok | n/a | n/a |
| `shadow_hand` | ok | ok | ok | FAIL | FAIL | FAIL | FAIL |
| `sharpa_wave_left` | ok | ok | ok | ok | ok | ok | pass |
| `sharpa_wave_right` | ok | ok | ok | ok | ok | ok | pass |
| `so_arm100` | ok | ok | ok | ok | ok | ok | pass |
| `unitree_dex3_1_left` | ok | n/a | ok | ok | ok | n/a | n/a |
| `ur10e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e` | ok | ok | ok | ok | ok | ok | pass |
| `xhand_right` | ok | n/a | ok | ok | ok | n/a | n/a |
| `yam` | ok | ok | ok | ok | ok | ok | pass |
| `z1` | ok | ok | ok | ok | ok | ok | pass |

Named failures (the first per robot and check):

- `aloha_agilex` / config: `no joint_limits for ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4', 'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint1', 'fr_joint2', 'fr_joint3', 'fr_joint4', 'fr_joint5', 'fr_joint6', 'fr_joint7', 'fr_`
- `arx_l5` / hold: `after 200 steps at the default pose: max joint deviation 0.0616 rad (joint7), base moved 0.0000 m`
- `cartpole` / hold: `could not build the robot on mujoco: FatalError: add_pair: broadphase buffer full`
- `g1_tracking` / config: `no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_shoulder_yaw_joint', 'left_wri`
- `g1_tracking` / assets: `['roboverse_data/unitree_description/mjcf/g1.xml'] not local and not fetched from the Hub (Exception)`
- `g1_tracking` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'`
- `g1_tracking` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'`
- `g1_tracking` / hold: `could not build the robot on mujoco: Exception: File roboverse_data/unitree_description/mjcf/g1.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/roboverse_da`
- `koch_1` / hold: `after 200 steps at the default pose: max joint deviation 0.2189 rad (pitch), base moved 0.0000 m`
- `openarm_bimanual_wuji` / assets: `['roboverse_data/robots/openarm_wuji/openarm_wuji.xml'] not local and not fetched from the Hub (Exception)`
- `openarm_bimanual_wuji` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_bimanual_wuji` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_bimanual_wuji` / hold: `could not build the robot on mujoco: Exception: File roboverse_data/robots/openarm_wuji/openarm_wuji.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/robover`
- `openarm_wuji` / assets: `['roboverse_data/robots/openarm_wuji/openarm_wuji.xml'] not local and not fetched from the Hub (Exception)`
- `openarm_wuji` / limits: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_wuji` / actuators: `MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'`
- `openarm_wuji` / hold: `could not build the robot on mujoco: Exception: File roboverse_data/robots/openarm_wuji/openarm_wuji.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/robover`
- `shadow_hand` / limits: `MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 `
- `shadow_hand` / actuators: `MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 `
- `shadow_hand` / hold: `could not build the robot on mujoco: AttributeError: Line 4: error while parsing element <option>: 'apirate' is not a valid attribute for <option>`

## Cross-engine parity

### Cross-engine parity: mujoco vs isaacsim (2026-11-28)

`python -m metasim robot parity <robot> --sims mujoco isaacsim`: the same canned joint trajectory on both engines, judged per joint. Source: `results/robots/2026-11-28-parity-mujoco-isaacsim/`.

| robot | run | control steps | peak speed | worst joint | max diff | verdict |
|---|---|---|---|---|---|---|
| `franka` | default | 400 | 0.63/s | `panda_joint1` | 0.0043 rad | pass |
| `franka` | full range | 400 | 1.52/s | `panda_joint6` | 0.0034 rad | pass |
| `gen3` | default | 400 | 0.63/s | `joint_4` | 0.0004 rad | pass |
| `h1` | default | 400 | 0.63/s | `right_knee` | 1.4243 rad | reported, not judged (floating base) |
| `h1_wrist` | default | 400 | 0.63/s | `right_knee` | 1.6754 rad | reported, not judged (floating base) |
| `ur10e` | default | 400 | 0.63/s | `shoulder_pan_joint` | 0.0361 rad | pass |
| `ur5e` | default | 400 | 0.63/s | `shoulder_pan_joint` | 0.0275 rad | pass |
| `z1` | default | 400 | 0.63/s | `joint1` | 0.0011 rad | pass |
