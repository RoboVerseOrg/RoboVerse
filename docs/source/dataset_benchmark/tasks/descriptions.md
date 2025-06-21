# Benchmark Descriptions

## Task List

RoboVerse also includes various predefined scenes that can be easily loaded into your simulation.

We are now actively collecting scenes from multiple sources and updating the existing scenes to match our newest configuration system. The length of the list and number of scenes will continuously increase. You are also welcomed to contribute your own scene to RoboVerse.

| Source | Path in Roboverse | Categories & Amounts                      |
| ------ | ------------------ | ----------------------- |
| [ARNOLD](#arnold) | `metasim/cfg/tasks/arnold` | House (6) |
| [CALVIN](#calvin) | `metasim/cfg/tasks/calvin` | CALVIN Table (4: Calvin Table A, B, C, D) |
| [DMControl](#dmcontrol) | `metasim/cfg/tasks/dmcontrol` | Locomotion (1) |
| Fetch | `metasim/cfg/tasks/fetch` | Manipulation (1) |
| [GAPartManip](#gapartmanip) | `metasim/cfg/tasks/gapartmanip` | Manipulation (2) |
| [GAPartNet](#gapartnet) | `metasim/cfg/tasks/gapartnet` | Manipulation (5) |
| GPT(GPT Generated Tasks) | `metasim/cfg/tasks/gpt` | Manipulation (1) |
| [GraspNet](#graspnet) | `metasim/cfg/tasks/graspnet` | Grasping (1) |
| [HumanoidBench](#humanoidbench) | `metasim/cfg/tasks/humanoidbench` | Humanoid (19) |
| [IsaacgymEnvs](#isaacgymenvs) | `metasim/cfg/tasks/isaacgym_envs` | Locomotion (1) Manipulation (1) |
| [LIBERO](#libero) | `metasim/cfg/tasks/libero` | Manipulation (10) |
| [Maniskill](#maniskill) | `metasim/cfg/tasks/maniskill` | Manipulation (7) |
| [MetaWorld](#metaworld) | `metasim/cfg/tasks/metaworld` | Manipulation (6) |
| [Open6Dor](#open6dor) | `metasim/cfg/tasks/open6dor` | Manipulation (68) |
| [RLAfford](#rlafford) | `metasim/cfg/tasks/rlafford` | Manipulation (1) |
| [RLBench](#rlbench) | `metasim/cfg/tasks/rlbench` | Manipulation (68) |
| [RoboSuite](#robosuite) | `metasim/cfg/tasks/robosuite` | Manipulation (7) |
| [SimplerEnv](#simplerenv) | `metasim/cfg/tasks/simpler_env` | Manipulation (1) |
| [UH1](#uh1) | `metasim/cfg/tasks/uh1` | Humanoid (1) |

### ARNOLD

[ARNOLD](https://arnold-benchmark.github.io/) is a benchmark for language-conditioned manipulation.
The benchmark uses motion planning and keypoints for robot manipulation tasks, focusing on fine-grained language understanding.
Tasks and Assets: We integrate six out of eight tasks from Arnold into RoboVerse: picking up objects, reorienting
objects, opening/closing drawers, and opening/closing cabinets.
Demonstrations: As the benchmark does not use trajectory-level demonstrations, we use motion planning for trajectory
generation to interpolate between keypoints

### CALVIN

[CALVIN](http://calvin.cs.uni-freiburg.de/) provides 6-hour teleopreation trajectories on 4 environments, each involve an articulated table with three blocks in blue, pink, or red.

**Demonstrations**: We migrate the demonstrations in all $4$ environments and transform the original assets (URDF for the table, and primitives for the cubes) into USD files with proper physics APIs.

**Success checkers**: We segment the trajectories according to the text annotations, which specified the task category (eg, PlaceInSlider), the text annotation (e.g., place the red block in the slider), and the timestamps of the demonstration segment. The states of the first frame is adopted as the scene initial states.

**Success checkers**: We carefully implement the success checkers according to the original implementation to make sure the failed executions can be filtered out. This is because the coarsely annotated timestamps in the dataset, which may cause the failed execution in part of the demonstrations.

### DMControl

[DMControl](https://deepmind.google/discover/blog/dm-control-software-and-tasks-for-continuous-control/) is Google DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.

We only adapted the Walker environment to RoboVerse.

### GAPartManip

[GAPartManip](https://arxiv.org/abs/2411.18276) offers a large-scale, part-oriented, scene-level dataset with annotations for actionable interaction poses. We utilize the mesh-level grasping pose annotations in this dataset to generate diverse demonstrations for articulated object manipulation.

**Tasks and Assets**: We currently implement two tasks: OpenBox and OpenToilet. For the OpenBox task, we collect 12 object assets from the Box category in the original dataset. For the OpenToilet task, we gather 30 objects from the Toilet category. We convert these assets into USD files with appropriate physics APIs to ensure compatibility with our simulation environment.

**Demonstrations**: We generate demonstrations for our tasks in simulation using motion planning with [CuRobo](https://curobo.org/). First, we filter potential grasping poses for the target object link by assessing their feasibility through motion planning. Specifically, we discard poses that the end-effector cannot reach or that would cause a collision between the robot and the object.

Next, we generate an end-effector pose trajectory to complete the task using heuristics. Based on the object's kinematic tree, we could define an ideal trajectory. We then apply motion planning to perform inverse kinematics, computing the corresponding joint poses of the robot along this trajectory. Finally, we execute the planned trajectory in simulation to verify task completion, saving successful trajectories as demonstrations. The entire demonstration generation process is conducted in IsaacSim.

**Success Checkers**: To determine task success, we require the manipulated object to be opened by at least 60 degrees for all tasks.

### GAPartNet

For tasks in [GAPartNet](https://pku-epic.github.io/GAPartNet/), we generate both motion planning and reinforcement learning trajectories. GAPartNet is implemented in IsaacGym  with various articulated objects. To integrate it into RoboVerse, we first align all articulated object initial states to the MetaSim format and convert the asset format to USD for compatibility across different simulators.

For trajectory generation:

(1) Motion Planning: GAPartNet introduces a part-centric manipulation approach. We roll out heuristics to generate manipulation trajectories, providing three demonstrations per part with different object and part initial states.

(2) Reinforcement Learning Rollout: The follow-up work, PartManip, proposes several reinforcement learning methods. We re-train all policies based on our robot setup and roll out trajectories for dataset collection.

With aligned task configurations, trajectories, and assets, we successfully adapt GAPartNet into RoboVerse.

### GraspNet

[GraspNet-1B](https://graspnet.net/) is a general object grasping dataset for predicting 6 DoF grasping pose given partial pointcloud input.
It contains 256 realworld tabletop scenes consists of total 88 different objects.
We carefully filter out 58 objects as our target grasping objects based on the availability of purchasing real items because we need to evaluate our policies to grasp them in the real world experiments.

To generate grasping demonstrations, we use [CuRobo](https://curobo.org/) as motion planner to generate robot end effector trajectories starting from a fixed initial pose and ending to an target object grasping pose.

The grasping pose is obtained from the grasping annotations used to train GraspNet.
We also randomized the object positions to generate more diverse layouts.
Finally, we validate the trajectories in our framework and filter out invalid ones by controlling robots to follow the generated grasping trajectories.

In the end, we successfully generated about 100k valid grasping trajectories.

### HumanoidBench

[HumanoidBench](https://humanoid-bench.github.io/) is the first-of-its-kind simulated humanoid robot benchmark, including 27 distinct whole-body control tasks, each of these presenting unique challenges, such as intricate long-horizon control and sophisticated coordination.

We included 19/27 tasks from the resource.

### IsaacgymEnvs

[IsaacgymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) provides Isaacgym based simulation environments. We adopted the Ant environment and allegrohand environment to RoboVerse.

### LIBERO

[LIBERO](https://libero-project.github.io/main.html) manages data loading and task execution through a combination of INIT(initialization files), BDDL (Behavior Description Definition Language), and HDF5 datasets. Specifically, the initialization files define scene layouts, object properties, and basic task goals; the BDDL format captures semantic details and object affordances; and the HDF5 files store structured data such as object positions and robot actions for dynamic retrieval at runtime.

To migrate a LIBERO task into MetaSim, we parse the relevant BDDL file to identify which objects are involved and what type of manipulation context is required. Then we get the robot and object initial states from the INIT files, followed by the corresponding robot actions from the HDF5 dataset. These elements are combined into our PKL file format while also recording the participating objects in our MetaCfg. This process ensures that all necessary components of a LIBERO task, initial states, and action data, are fully translated and ready for execution in MetaSim.

We further augment the data by randomly sampling initial positions around each LIBERO demonstration, thus increasing the effective number of demos well beyond the original 50 per task. The spatial sampling range is dynamically chosen based on the task context and object dimensions, ensuring that the augmented configurations remain physically plausible.

### Maniskill

[Maniskill](https://www.maniskill.ai/) provides a series of robotic manipulation tasks under single-arm or dual-arm settings.

**Tasks and assets**: We migrate basic single-arm tasks and demonstrations to RoboVerse, including the pick-and-place tasks like PickCube and PickSingleYCB, as well as the insertion tasks like PegInsertionSide and PlugCharger. The corresponding assets are manually crafted with primitives or process from the mesh files, with proper physics API set up.

**Demonstrations**: For each task, a great number of demonstration trajectories are available in the released data. Noteworthy, the data does not come with the initial scene states, which are obtained by replaying the demonstrations within the SAPIEN simulator. With the specified seed set, the states are recovered by the random samplers.The success checkers are implemented according to the task designs.

### Metaworld

[Metaworld](https://meta-world.github.io/) is a widely used benchmark for multi-task and meta-reinforcement learning, comprising 50 distinct tabletop robotic manipulation tasks involving a Sawyer robot.

**Tasks and Assets**: We integrate five representative tasks into RoboVerse: drawer open, drawer close, door close, window open, and window close. The corresponding assets are manually converted from MJCF to USD files with appropriate physics APIs.

**Demonstrations**: As the benchmark does not provide demonstrations, we generate trajectories for each task by rolling out reinforcement learning policies from [Text2reward](https://text-to-reward.github.io/).

### Open6Dor

[Open6Dor](https://pku-epic.github.io/Open6DOR/) is a benchmark for open-instruction 6-DoF object rearrangement tasks, which requires embodied agents to move the target objects according to open instructions that specify its 6-DoF pose.

**Tasks and Assets**: The synthetic object dataset comprises $200+$ items spanning 70+ distinct categories. Originally derived from YCB and Objaverse-XL, the objects are carefully filtered and scaled using a standardized format of mesh representation.
Overall, the Open6DOR Benchmark consists of
5k+ tasks, divided into the position-track, rotation-track, and 6-DoF-track, each providing manually configured tasks along with
comprehensive and quantitative 3D annotations.

**Success checkers**: We determine success by comparing the target object's final pose with the annotated ground-truth pose range.

### RLAfford

[RLAfford](https://sites.google.com/view/rlafford/) investigates the generalization ability of Deep Reinforcement Learning models on articulated object manipulation tasks with the presence of a computer vision model that is co-trained with it in an end-to-end
manner. This work provided a dataset of articulated objects and $8$ tasks for benchmarking.

In Roboverse, we have adapted $4$ tasks (open cabinet, open drawer, close cabinet,
close drawer) and in total $40$k trajectories from RLAfford. Currently only the open cabinet task is released.

In the task adaptation, we included $40$ articulated objects from the RLAfford
dataset, and uses the same robot description file from RLAfford. Then we record
$1000$ trajectories for each object in its corresponding task.

The trajectory recording is achieved with several hooks we inserted into the
original RLAfford codebase. The hooks are used to extract and maintain the recordings
at different stages of simulation. We evaluated the released RLAfford model with
hook-inserted scripts. In the initialization stage, objects and robots are
initialized with randomization, their pose, and DoF information are recorded. For
each simulation step, the DoF position information of objects and robots is
recorded in the trajectories. In the end, for each object, a separate trajectory
file of $1000$ different trajectories is saved in the RoboVerse supported format.

### RLBench

[RLBench](https://github.com/stepjam/RLBench) is a large-scale benchmark and learning environment for robotic manipulation, featuring $100$ diverse, hand-designed tasks ranging in complexity, from simple actions like reaching to multi-stage tasks like opening an oven and placing a tray inside. Each task includes an infinite supply of demonstrations generated via waypoint-based motion planning.

**Tasks and assets**: We roll out ${\sim}2K$ trajectories in RLBench for each task, and migrate them to RoboVerse.

### RoboSuite

[RoboSuite](https://robosuite.ai/) provides a suite of task environments for robotic manipulation, built on the MuJoCo physics engine. Each task is implemented as a separate class, with most configuration details embedded in the source code. Based on these environments, MimicGen offers thousands of demonstrations, serving as a widely used benchmark for imitation learning.


**Tasks and Assets**: For tasks with separate object description files (MJCF), we directly migrate the corresponding assets through our Asset Conversion pipeline. However, some tasks contain hard-coded assets within the source code, such as a hammer composed of multiple cubes, cylinders and other primitives with carefully designed relative poses. To integrate these tasks, we will manually reconstruct the assets within our framework. We also argue that hard-coded asset and task definitions, as opposed to modular task descriptions, are not scalable for future robotic task benchmarking.

**Demonstrations**: We convert MimicGen demonstrations into our format. Specifically, we transform the robot actions from 6-DoF Cartesian space representations to joint space. Additionally, the state of the first frame is adopted as the initial scene state.

**Success Checkers**: We meticulously implement success checkers based on the original definitions to ensure failed executions are effectively filtered out.

### SimplerEnv

[SimplerEnv](https://simpler-env.github.io/) is a set of tasks and methods designed to do trustworthy benchmarking in simulation for manipulation policies that can reflect the real-world success rate.

There are in total $25$ different tasks in SimplerEnv. We ignore all tasks that are just a subset of another task and migrated in total $6$ tasks and $52$ object assets to RoboVerse. The tasks all use Google Robot. Currently only one task is uploaded.

SimplerEnv provided some controller models trained with RT-1 and RT-X dataset. We did not use the trajectories from the dataset directly because some environmental settings are different from the environments from SimplerEnv. We used the trained model to collect trajectories. Hooks are inserted into the original SimplerEnv codebase to extract and maintain the recordings at different stages of simulation. We then rollout the model trained with RT-1 dataset on each task to collect the trajectories.

### UH1

[UH1](https://arxiv.org/abs/2412.14172) is a large-scale dataset of
over 20 million humanoid robot poses with corresponding
text-based motion descriptions. We currently included one set of trajectory into RoboVerse.
