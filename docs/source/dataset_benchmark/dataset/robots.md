# Robots

Generated from `roboverse_pack/robots` by `generate_robot_docs.py`; do not edit by hand. A robot is listed
with the asset formats its config declares; `tools/assets/check_manifest.py` verifies on every pull request
that each declared file exists on the Hub. Robots whose assets did not exist were retired on 2026-11-28
(`roboverse_pack/tasks/RETIRED_TASKS.md`).

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
