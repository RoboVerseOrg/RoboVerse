# Quick Start Tutorials

This section provides step-by-step tutorials to help you get started with MetaSim. Each tutorial builds on previous concepts, guiding you from basic simulation setup to advanced features.

---

## Learning Path

We recommend following the tutorials in order for the best learning experience:

### Beginner Level

| Tutorial | Description | Time |
|----------|-------------|------|
| [Guide](guide) | Overview and basic concepts | 10 min |
| [Static Scene](0_static_scene) | Load and visualize a simulation scene | 15 min |
| [Control Robot](1_control_robot) | Send commands to control robot joints | 20 min |
| [Add New Robot](2_add_new_robot) | Import and configure custom robots | 30 min |

### Intermediate Level

| Tutorial | Description | Time |
|----------|-------------|------|
| [Parallel Environments](3_parallel_envs) | Run multiple environments simultaneously | 20 min |
| [Motion Planning](4_motion_planning) | Plan collision-free robot trajectories | 30 min |
| [Hybrid Simulation](5_hybrid_sim) | Combine different simulation backends | 25 min |
| [Advanced Rendering](6_advanced_rendering) | Customize visual rendering options | 20 min |

### Advanced Level

| Tutorial | Description | Time |
|----------|-------------|------|
| [Collect Demonstrations](7_collect_demo) | Record expert demonstrations | 25 min |
| [Replay Demonstrations](8_replay_demo) | Playback and analyze recordings | 15 min |
| [Configure Tasks](9_cfg_task) | Create custom task configurations | 30 min |
| [Mount Camera](10_mount_camera) | Attach cameras to robots and objects | 20 min |

### Expert Level

| Tutorial | Description | Time |
|----------|-------------|------|
| [Domain Randomization](12_domain_randomization) | Randomize physics and visuals | 30 min |
| [Teleoperation](13_teleoperate) | Control robots with external devices | 25 min |
| [Real Assets](14_real_asset) | Use real-world scanned assets | 30 min |
| [GS Background](15_gs_background) | Gaussian splatting backgrounds | 25 min |
| [EmbodiedGen Layout](16_embodiedgen_layout) | Generate scene layouts automatically | 30 min |
| [Rerun Visualization](17_rerun_visualization) | Advanced visualization with Rerun | 20 min |

---

## Prerequisites

Before starting, ensure you have:

- Completed the [installation](../installation) process
- Basic familiarity with Python and NumPy
- (Optional) GPU with CUDA support for accelerated simulation

---

## Tutorial Format

Each tutorial follows a consistent structure:

1. **Objective**: What you'll learn
2. **Prerequisites**: Required knowledge and setup
3. **Step-by-Step Guide**: Detailed instructions with code
4. **Expected Output**: What you should see
5. **Next Steps**: Where to go from here

---

```{toctree}
:titlesonly:
:hidden:

guide
0_static_scene
1_control_robot
2_add_new_robot
3_parallel_envs
4_motion_planning
5_hybrid_sim
6_advanced_rendering
7_collect_demo
8_replay_demo
9_cfg_task
10_mount_camera
12_domain_randomization
13_teleoperate
14_real_asset
15_gs_background
16_embodiedgen_layout
17_rerun_visualization
```
