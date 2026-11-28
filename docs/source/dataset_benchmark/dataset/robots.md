# Robots

Generated from `roboverse_pack/robots` and `results/robots/` by `generate_robot_docs.py`; do not edit by hand.
A robot is listed with the asset formats its config declares; `tools/assets/check_manifest.py` verifies on
every pull request that each declared file exists on the Hub. Robots whose assets did not exist were retired
on 2026-11-28 (`roboverse_pack/tasks/RETIRED_TASKS.md`). The self-check and parity tables come from the newest
result files per simulator; how to add a robot and pass them is in [Add a robot](../../start/add_a_robot.md).

## Configs

| config module | class | `name` | joints | asset formats |
|---|---|---|---|---|
| `aloha` | `AlohaCfg` | `aloha` | 16 |  |
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
| `h1_2_without_hand_cfg` | `H12WithoutHandCfg` | `h1_2_without_hand` | 27 | Isaac Sim |
| `h1_body_collision` | `H1BodyCollisionCfg` | `h1` | 26 | MuJoCo, Isaac Sim, URDF, MJX |
| `h1_cfg` | `H1Cfg` | `h1` | 26 | MuJoCo, Isaac Sim, URDF, MJX |
| `h1_hand_cfg` | `H1HandCfg` | `h1_hand` | 76 | MuJoCo, MJX |
| `h1_simple_hand_cfg` | `H1SimpleHandCfg` | `h1_simple_hand` | 52 | MuJoCo, MJX |
| `h1_wrist_cfg` | `H1WristCfg` | `h1_wrist` | 19 | MuJoCo, Isaac Sim, URDF, MJX |
| `iiwa14_cfg` | `Iiwa14Cfg` | `iiwa14` | 7 | MuJoCo, Isaac Sim, URDF, MJX |
| `iiwa_cfg` | `IiwaCfg` | `iiwa` | 9 | Isaac Sim |
| `kinova_gen3_robotiq_2f85` | `KinovaGen3Robotiq2f85Cfg` | `kinova_gen3_robotiq_2f85` | 9 | Isaac Sim |
| `koch_mjcf_cfg` | `KochCfg` | `koch_1` | 6 | MuJoCo, MJX |
| `lite6_mjcf_cfg` | `Lite6Cfg` | `lite6` | 6 | MuJoCo, MJX |
| `mjlab_cartpole_cfg` | `MjlabCartpoleCfg` | `cartpole` | 2 | MuJoCo, Isaac Sim, URDF, MJX |
| `mjlab_g1_cfg` | `MjlabG1Cfg` | `g1` | 29 | MuJoCo, Isaac Sim, URDF, MJX |
| `mjlab_go1_cfg` | `MjlabGo1Cfg` | `go1` | 12 | MuJoCo, Isaac Sim, URDF, MJX |
| `mjlab_yam_cfg` | `MjlabYamCfg` | `yam` | 8 | MuJoCo, Isaac Sim, URDF, MJX |
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
| `ur5e_2f85_cfg` | `Ur5E2F85Cfg` | `ur5e_2f85` | 12 | Isaac Sim |
| `ur5e_cfg` | `Ur5ECfg` | `ur5e` | 6 | MuJoCo, Isaac Sim, URDF, MJX |
| `vega_cfg` | `VegaCfg` | `vega` | 45 | Isaac Sim, URDF |
| `xhand_cfg` | `XhandRightCfg` | `xhand_right` | 12 | URDF |
| `yam_mjcf_cfg` | `YamCfg` | `yam` | 8 | MuJoCo, MJX |
| `z1_cfg` | `Z1Cfg` | `z1` | 6 | MuJoCo, Isaac Sim, URDF, MJX |

49 robot configs.

## Self-check

### Self-check on isaacsim (2026-11-28)

`python -m metasim robot check --all --sim isaacsim`: 42 robots, 9 pass, 33 fail, 0 n/a (no asset declared for this backend). Source: `results/robots/2026-11-28-isaacsim-check/robots.json`.

| robot | config | assets | kinematics | limits | actuators | hold | verdict |
|---|---|---|---|---|---|---|---|
| `aloha` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `aloha_agilex` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `arx_l5` | ok | FAIL | ok | ok | FAIL | FAIL | FAIL |
| `cartpole` | ok | ok | ok | FAIL | FAIL | FAIL | FAIL |
| `fetch` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `franka` | ok | ok | ok | ok | ok | ok | pass |
| `franka_with_gripper_extension` | FAIL | ok | FAIL | ok | ok | FAIL | FAIL |
| `g1` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `g1_dof12` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof23` | ok | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `g1_dof27` | ok | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `g1_dof29` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof29_dex3` | ok | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `g1_tracking` | FAIL | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `gen3` | ok | ok | ok | ok | ok | ok | pass |
| `go1` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `google_robot_static` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `h1` | ok | ok | ok | ok | ok | ok | pass |
| `h1_2_without_hand` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `h1_hand` | FAIL | FAIL | ok | ok | FAIL | FAIL | FAIL |
| `h1_simple_hand` | FAIL | FAIL | ok | FAIL | ok | FAIL | FAIL |
| `h1_wrist` | ok | ok | ok | ok | ok | ok | pass |
| `iiwa` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `iiwa14` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `kinova_gen3_robotiq_2f85` | FAIL | ok | FAIL | ok | ok | FAIL | FAIL |
| `koch_1` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `lite6` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `openarm_bimanual_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `openarm_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `sawyer` | FAIL | ok | FAIL | ok | ok | FAIL | FAIL |
| `shadow_hand` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `sharpa_wave_left` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `sharpa_wave_right` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `so_arm100` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `unitree_dex3_1_left` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `ur10e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e_2f85` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `vega` | FAIL | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `xhand_right` | ok | FAIL | FAIL | ok | ok | FAIL | FAIL |
| `yam` | ok | FAIL | ok | ok | ok | FAIL | FAIL |
| `z1` | ok | ok | ok | ok | ok | ok | pass |

Named failures (the first per robot and check):

- `aloha` / config: no default position for ['left/elbow', 'left/forearm_roll', 'left/left_finger', 'left/right_finger', 'left/shoulder', 'left/waist', 'left/wrist_angle', 'left/wrist_rotate', 'right/elbow', 'right/forearm_roll', 'right/lef
- `aloha` / assets: no usd_path declared for isaacsim
- `aloha` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `aloha` / hold: not attempted: a cheaper check already failed
- `aloha_agilex` / config: no joint_limits for ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4', 'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint1', 'fr_joint2', 'fr_joint3', 'fr_joint4', 'fr_joint5', 'fr_joint6', 'fr_joint7', 'fr_
- `aloha_agilex` / assets: no usd_path declared for isaacsim
- `aloha_agilex` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `aloha_agilex` / hold: not attempted: a cheaper check already failed
- `arx_l5` / assets: no usd_path declared for isaacsim
- `arx_l5` / actuators: 1 actuator(s) not named after their joint (the handler would add a second servo): 'gripper' drives 'joint7'
- `arx_l5` / hold: not attempted: a cheaper check already failed
- `cartpole` / limits: 1 of 1 limited joints disagree with the MJCF: slider cfg [-2.000, 2.000] vs mjcf [-1.800, 1.800]
- `cartpole` / actuators: 1 actuator(s) not named after their joint (the handler would add a second servo): 'slide' drives 'slider'
- `cartpole` / hold: not attempted: a cheaper check already failed
- `fetch` / config: no default position for ['elbow_flex_joint', 'forearm_roll_joint', 'head_pan_joint', 'head_tilt_joint', 'l_gripper_finger_joint', 'l_wheel_joint', 'r_gripper_finger_joint', 'r_wheel_joint', 'shoulder_lift_joint', 'should
- `fetch` / assets: no usd_path declared for isaacsim
- `fetch` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `fetch` / hold: not attempted: a cheaper check already failed
- `franka_with_gripper_extension` / config: no default position for ['panda_finger_joint1', 'panda_finger_joint2', 'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
- `franka_with_gripper_extension` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `franka_with_gripper_extension` / hold: not attempted: a cheaper check already failed
- `g1` / hold: child exited 1 without a report: 2026-11-28 09:14:10.974 +00:00 / ERROR    / metasim.sim.isaacsim.isaacsim:close:705 - closing Isaac Sim while ValueError propagates; the process exits 1
- `g1_dof23` / assets: no usd_path declared for isaacsim
- `g1_dof23` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `g1_dof23` / hold: not attempted: a cheaper check already failed
- `g1_dof27` / assets: no usd_path declared for isaacsim
- `g1_dof27` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `g1_dof27` / hold: not attempted: a cheaper check already failed
- `g1_dof29_dex3` / assets: no usd_path declared for isaacsim
- `g1_dof29_dex3` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `g1_dof29_dex3` / hold: not attempted: a cheaper check already failed
- `g1_tracking` / config: no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_shoulder_yaw_joint', 'left_wri
- `g1_tracking` / assets: ['/workspace/roboverse_maint/RoboVerse/roboverse_data/unitree_description/usd/g1/g1.usd'] not local and not fetched from the Hub (Exception)
- `g1_tracking` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'
- `g1_tracking` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'
- `g1_tracking` / hold: not attempted: a cheaper check already failed
- `go1` / hold: child exited 1 without a report: 2026-11-28 09:15:14.432 +00:00 / ERROR    / metasim.sim.isaacsim.isaacsim:close:705 - closing Isaac Sim while ValueError propagates; the process exits 1
- `google_robot_static` / config: no joint_limits for ['joint_bicep', 'joint_elbow', 'joint_finger_left', 'joint_finger_right', 'joint_forearm', 'joint_gripper', 'joint_head_pan', 'joint_head_tilt', 'joint_shoulder', 'joint_torso', 'joint_wrist']; no def
- `google_robot_static` / assets: no usd_path declared for isaacsim
- `google_robot_static` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `google_robot_static` / hold: not attempted: a cheaper check already failed
- `h1_2_without_hand` / config: no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left
- `h1_2_without_hand` / assets: ['roboverse_data_release/robots/h1_2_without_hand/usd/h1_2_without_hand.usd'] not local and not fetched from the Hub (ValueError)
- `h1_2_without_hand` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `h1_2_without_hand` / hold: not attempted: a cheaper check already failed
- `h1_hand` / config: no joint_limits for ['lh_FFJ1', 'lh_FFJ2', 'lh_FFJ3', 'lh_FFJ4', 'lh_LFJ1', 'lh_LFJ2', 'lh_LFJ3', 'lh_LFJ4', 'lh_LFJ5', 'lh_MFJ1', 'lh_MFJ2', 'lh_MFJ3', 'lh_MFJ4', 'lh_RFJ1', 'lh_RFJ2', 'lh_RFJ3', 'lh_RFJ4', 'lh_THJ1', '
- `h1_hand` / assets: no usd_path declared for isaacsim
- `h1_hand` / actuators: 48 actuator(s) not named after their joint (the handler would add a second servo): 'lh_A_WRJ2' drives 'lh_WRJ2'; 'lh_A_WRJ1' drives 'lh_WRJ1'; 'lh_A_THJ5' drives 'lh_THJ5'; 'lh_A_THJ4' drives 'lh_THJ4'
- `h1_hand` / hold: not attempted: a cheaper check already failed
- `h1_simple_hand` / config: no default position for ['L_index_intermediate', 'L_index_proximal', 'L_middle_intermediate', 'L_middle_proximal', 'L_pinky_intermediate', 'L_pinky_proximal', 'L_ring_intermediate', 'L_ring_proximal', 'L_thumb_distal', '
- `h1_simple_hand` / assets: no usd_path declared for isaacsim
- `h1_simple_hand` / limits: 26 of 45 limited joints disagree with the MJCF: left_hand cfg [-6.000, 6.000] vs mjcf [-3.054, 3.054]; right_hand cfg [-6.000, 6.000] vs mjcf [-3.054, 3.054]; L_index_proximal cfg [-1.000, 1.000] vs mjcf [0.000, 1.700]; 
- `h1_simple_hand` / hold: not attempted: a cheaper check already failed
- `iiwa` / config: no joint_limits for ['iiwa7_joint_1', 'iiwa7_joint_2', 'iiwa7_joint_3', 'iiwa7_joint_4', 'iiwa7_joint_5', 'iiwa7_joint_6', 'iiwa7_joint_7', 'panda_finger_joint1', 'panda_finger_joint2']; no default position for ['iiwa7_j
- `iiwa` / assets: ['data_isaaclab/robots/iiwa/iiwa.usd'] not local and not fetched from the Hub (ValueError)
- `iiwa` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `iiwa` / hold: not attempted: a cheaper check already failed
- `iiwa14` / hold: child exited 1 without a report: 2026-11-28 09:15:59.293 +00:00 / ERROR    / metasim.sim.isaacsim.isaacsim:close:705 - closing Isaac Sim while ValueError propagates; the process exits 1
- `kinova_gen3_robotiq_2f85` / config: no default position for ['finger_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'left_outer_finger_joint', 'right_outer_finger_joint']
- `kinova_gen3_robotiq_2f85` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `kinova_gen3_robotiq_2f85` / hold: not attempted: a cheaper check already failed
- `koch_1` / assets: no usd_path declared for isaacsim
- `koch_1` / hold: not attempted: a cheaper check already failed
- `lite6` / assets: no usd_path declared for isaacsim
- `lite6` / hold: not attempted: a cheaper check already failed
- `openarm_bimanual_wuji` / assets: ['roboverse_data/robots/openarm_wuji/openarm_wuji.usd'] not local and not fetched from the Hub (Exception)
- `openarm_bimanual_wuji` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_bimanual_wuji` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_bimanual_wuji` / hold: not attempted: a cheaper check already failed
- `openarm_wuji` / assets: ['roboverse_data/robots/openarm_wuji/openarm_wuji.usd'] not local and not fetched from the Hub (Exception)
- `openarm_wuji` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_wuji` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_wuji` / hold: not attempted: a cheaper check already failed
- `sawyer` / config: no default position for ['head_pan', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint', 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
- `sawyer` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `sawyer` / hold: not attempted: a cheaper check already failed
- `shadow_hand` / assets: no usd_path declared for isaacsim
- `shadow_hand` / limits: MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 
- `shadow_hand` / actuators: MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 
- `shadow_hand` / hold: not attempted: a cheaper check already failed
- `sharpa_wave_left` / assets: no usd_path declared for isaacsim
- `sharpa_wave_left` / hold: not attempted: a cheaper check already failed
- `sharpa_wave_right` / assets: no usd_path declared for isaacsim
- `sharpa_wave_right` / hold: not attempted: a cheaper check already failed
- `so_arm100` / assets: no usd_path declared for isaacsim
- `so_arm100` / hold: not attempted: a cheaper check already failed
- `unitree_dex3_1_left` / config: no default position for ['left_hand_index_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint']
- `unitree_dex3_1_left` / assets: no usd_path declared for isaacsim
- `unitree_dex3_1_left` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `unitree_dex3_1_left` / hold: not attempted: a cheaper check already failed
- `ur5e_2f85` / config: no default position for ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
- `ur5e_2f85` / assets: ['data_isaaclab/robots/UniversalRobots/ur5e/ur5e_2f85_fix.usd'] not local and not fetched from the Hub (ValueError)
- `ur5e_2f85` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `ur5e_2f85` / hold: not attempted: a cheaper check already failed
- `vega` / config: default outside limits for ['torso_j1']
- `vega` / assets: ['roboverse_pack/robots/robots_vega/humanoid/vega_1/vega/vega.usd'] not local and not fetched from the Hub (ValueError)
- `vega` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `vega` / hold: not attempted: a cheaper check already failed
- `xhand_right` / assets: no usd_path declared for isaacsim
- `xhand_right` / kinematics: no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model
- `xhand_right` / hold: not attempted: a cheaper check already failed
- `yam` / assets: no usd_path declared for isaacsim
- `yam` / hold: not attempted: a cheaper check already failed

### Self-check on mujoco (2026-11-28)

`python -m metasim robot check --all --sim mujoco`: 42 robots, 17 pass, 9 fail, 16 n/a (no asset declared for this backend). Source: `results/robots/2026-11-28-mujoco-check/robots.json`.

| robot | config | assets | kinematics | limits | actuators | hold | verdict |
|---|---|---|---|---|---|---|---|
| `aloha` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `aloha_agilex` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `arx_l5` | ok | ok | ok | ok | FAIL | FAIL | FAIL |
| `cartpole` | ok | ok | ok | FAIL | FAIL | FAIL | FAIL |
| `fetch` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `franka` | ok | ok | ok | ok | ok | ok | pass |
| `franka_with_gripper_extension` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `g1` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof12` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof23` | ok | n/a | n/a | ok | ok | n/a | n/a |
| `g1_dof27` | ok | n/a | n/a | ok | ok | n/a | n/a |
| `g1_dof29` | ok | ok | ok | ok | ok | ok | pass |
| `g1_dof29_dex3` | ok | n/a | n/a | ok | ok | n/a | n/a |
| `g1_tracking` | FAIL | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `gen3` | ok | ok | ok | ok | ok | ok | pass |
| `go1` | ok | ok | ok | ok | ok | ok | pass |
| `google_robot_static` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `h1` | ok | ok | ok | ok | ok | ok | pass |
| `h1_2_without_hand` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `h1_hand` | FAIL | ok | ok | ok | FAIL | FAIL | FAIL |
| `h1_simple_hand` | FAIL | ok | ok | FAIL | ok | ok | FAIL |
| `h1_wrist` | ok | ok | ok | ok | ok | ok | pass |
| `iiwa` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `iiwa14` | ok | ok | ok | ok | ok | ok | pass |
| `kinova_gen3_robotiq_2f85` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `koch_1` | ok | ok | ok | ok | ok | FAIL | FAIL |
| `lite6` | ok | ok | ok | ok | ok | ok | pass |
| `openarm_bimanual_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `openarm_wuji` | ok | FAIL | ok | FAIL | FAIL | FAIL | FAIL |
| `sawyer` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `shadow_hand` | ok | ok | ok | FAIL | FAIL | FAIL | FAIL |
| `sharpa_wave_left` | ok | ok | ok | ok | ok | ok | pass |
| `sharpa_wave_right` | ok | ok | ok | ok | ok | ok | pass |
| `so_arm100` | ok | ok | ok | ok | ok | ok | pass |
| `unitree_dex3_1_left` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `ur10e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e` | ok | ok | ok | ok | ok | ok | pass |
| `ur5e_2f85` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `vega` | FAIL | n/a | n/a | ok | ok | n/a | n/a |
| `xhand_right` | ok | n/a | n/a | ok | ok | n/a | n/a |
| `yam` | ok | ok | ok | ok | ok | ok | pass |
| `z1` | ok | ok | ok | ok | ok | ok | pass |

Named failures (the first per robot and check):

- `aloha` / config: no default position for ['left/elbow', 'left/forearm_roll', 'left/left_finger', 'left/right_finger', 'left/shoulder', 'left/waist', 'left/wrist_angle', 'left/wrist_rotate', 'right/elbow', 'right/forearm_roll', 'right/lef
- `aloha_agilex` / config: no joint_limits for ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4', 'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint1', 'fr_joint2', 'fr_joint3', 'fr_joint4', 'fr_joint5', 'fr_joint6', 'fr_joint7', 'fr_
- `arx_l5` / actuators: 1 actuator(s) not named after their joint (the handler would add a second servo): 'gripper' drives 'joint7'
- `arx_l5` / hold: could not build the robot on mujoco: ValueError: Robot 'arx_l5': actuators ['gripper'] are not named after a joint of the robot (joints: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', '
- `cartpole` / limits: 1 of 1 limited joints disagree with the MJCF: slider cfg [-2.000, 2.000] vs mjcf [-1.800, 1.800]
- `cartpole` / actuators: 1 actuator(s) not named after their joint (the handler would add a second servo): 'slide' drives 'slider'
- `cartpole` / hold: could not build the robot on mujoco: FatalError: add_pair: broadphase buffer full
- `fetch` / config: no default position for ['elbow_flex_joint', 'forearm_roll_joint', 'head_pan_joint', 'head_tilt_joint', 'l_gripper_finger_joint', 'l_wheel_joint', 'r_gripper_finger_joint', 'r_wheel_joint', 'shoulder_lift_joint', 'should
- `franka_with_gripper_extension` / config: no default position for ['panda_finger_joint1', 'panda_finger_joint2', 'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
- `g1_tracking` / config: no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_shoulder_yaw_joint', 'left_wri
- `g1_tracking` / assets: ['roboverse_data/unitree_description/mjcf/g1.xml'] not local and not fetched from the Hub (Exception)
- `g1_tracking` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'
- `g1_tracking` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/unitree_description/mjcf/g1.xml'
- `g1_tracking` / hold: could not build the robot on mujoco: Exception: File roboverse_data/unitree_description/mjcf/g1.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/roboverse_da
- `google_robot_static` / config: no joint_limits for ['joint_bicep', 'joint_elbow', 'joint_finger_left', 'joint_finger_right', 'joint_forearm', 'joint_gripper', 'joint_head_pan', 'joint_head_tilt', 'joint_shoulder', 'joint_torso', 'joint_wrist']; no def
- `h1_2_without_hand` / config: no default position for ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left
- `h1_hand` / config: no joint_limits for ['lh_FFJ1', 'lh_FFJ2', 'lh_FFJ3', 'lh_FFJ4', 'lh_LFJ1', 'lh_LFJ2', 'lh_LFJ3', 'lh_LFJ4', 'lh_LFJ5', 'lh_MFJ1', 'lh_MFJ2', 'lh_MFJ3', 'lh_MFJ4', 'lh_RFJ1', 'lh_RFJ2', 'lh_RFJ3', 'lh_RFJ4', 'lh_THJ1', '
- `h1_hand` / actuators: 48 actuator(s) not named after their joint (the handler would add a second servo): 'lh_A_WRJ2' drives 'lh_WRJ2'; 'lh_A_WRJ1' drives 'lh_WRJ1'; 'lh_A_THJ5' drives 'lh_THJ5'; 'lh_A_THJ4' drives 'lh_THJ4'
- `h1_hand` / hold: could not build the robot on mujoco: ValueError: Robot 'h1_hand': the MJCF roboverse_data/robots/h1_hand/mjcf/h1_hand.xml ships no actuator for joint 'lh_WRJ2' and its BaseActuatorCfg sets no stiffness, so the
- `h1_simple_hand` / config: no default position for ['L_index_intermediate', 'L_index_proximal', 'L_middle_intermediate', 'L_middle_proximal', 'L_pinky_intermediate', 'L_pinky_proximal', 'L_ring_intermediate', 'L_ring_proximal', 'L_thumb_distal', '
- `h1_simple_hand` / limits: 26 of 45 limited joints disagree with the MJCF: left_hand cfg [-6.000, 6.000] vs mjcf [-3.054, 3.054]; right_hand cfg [-6.000, 6.000] vs mjcf [-3.054, 3.054]; L_index_proximal cfg [-1.000, 1.000] vs mjcf [0.000, 1.700]; 
- `iiwa` / config: no joint_limits for ['iiwa7_joint_1', 'iiwa7_joint_2', 'iiwa7_joint_3', 'iiwa7_joint_4', 'iiwa7_joint_5', 'iiwa7_joint_6', 'iiwa7_joint_7', 'panda_finger_joint1', 'panda_finger_joint2']; no default position for ['iiwa7_j
- `kinova_gen3_robotiq_2f85` / config: no default position for ['finger_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'left_outer_finger_joint', 'right_outer_finger_joint']
- `koch_1` / hold: after 200 steps at the default pose: max joint deviation 0.2189 rad (pitch), base moved 0.0000 m
- `openarm_bimanual_wuji` / assets: ['roboverse_data/robots/openarm_wuji/openarm_wuji.xml'] not local and not fetched from the Hub (Exception)
- `openarm_bimanual_wuji` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_bimanual_wuji` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_bimanual_wuji` / hold: could not build the robot on mujoco: Exception: File roboverse_data/robots/openarm_wuji/openarm_wuji.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/robover
- `openarm_wuji` / assets: ['roboverse_data/robots/openarm_wuji/openarm_wuji.xml'] not local and not fetched from the Hub (Exception)
- `openarm_wuji` / limits: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_wuji` / actuators: MJCF did not load: ValueError: ParseXML: Error opening file 'roboverse_data/robots/openarm_wuji/openarm_wuji.xml'
- `openarm_wuji` / hold: could not build the robot on mujoco: Exception: File roboverse_data/robots/openarm_wuji/openarm_wuji.xml neither exists in the local directory nor exists in the HuggingFace data sources ('RoboVerseOrg/robover
- `sawyer` / config: no default position for ['head_pan', 'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint', 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
- `shadow_hand` / limits: MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 
- `shadow_hand` / actuators: MJCF did not load: ValueError: XML Error: Schema violation: unrecognized attribute: 'apirate' Element 'option', line 4 
- `shadow_hand` / hold: could not build the robot on mujoco: AttributeError: Line 4: error while parsing element <option>: 'apirate' is not a valid attribute for <option>
- `unitree_dex3_1_left` / config: no default position for ['left_hand_index_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint']
- `ur5e_2f85` / config: no default position for ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
- `vega` / config: default outside limits for ['torso_j1']

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
