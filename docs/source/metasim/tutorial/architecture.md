# RoboVerse Project Architecture

## Overview

RoboVerse is a unified framework for robotic simulation and learning. It spans from multi-simulator abstraction, task systems, and scenario management to reinforcement learning (RL) and imitation learning (IL) modules. The design goal is to provide a highly modular, reusable, and extensible ecosystem that allows researchers and developers to quickly build, migrate, and evaluate diverse robotic learning tasks.

Project structure:

```
RoboVerse/
├── metasim/                    # Core simulation framework
│   ├── sim/                    # Simulator abstraction layer
│   ├── task/                   # Task system
│   ├── scenario/               # Scenario management
│   ├── utils/                  # Utility library
│   ├── queries/                # Query system
│   ├── example/                # Example code
│   └── tests/                  # Test code
│
├── roboverse_learn/            # Machine learning module
│   ├── il/                     # Imitation learning
│   └── rl/                     # Reinforcement learning
│
├── roboverse_pack/             # Pre-configured packages
│   ├── robots/                 # Robot configurations
│   ├── scenes/                 # Scene configurations
│   └── tasks/                  # Task configurations
│
├── get_started/                # Quick start examples
│   ├── motion_planning/        # Motion planning examples
│   ├── rl/                     # RL examples
│   └── dexhands/               # Dexterous hands examples
│
└── scripts/                    # Utility scripts
    ├── conversion/             # Format conversion
    ├── advanced/               # Advanced features
    ├── statistics/             # Data statistics
    └── mesh_tools/             # Mesh processing tools
```

---

## Module Description

### 1. **metasim - Core Simulation Framework**

Metasim is the core layer of RoboVerse, responsible for providing unified abstractions across different simulators and managing simulation states. Its goal is to shield upper layers from simulator-specific differences and expose a consistent state interface.

* **sim/**: Provides a unified abstraction layer for 11 simulators. Each simulator defines a handler (e.g., `MJXHandler`, `IsaacHandler`) that inherits from `BaseSimHandler` and is responsible for:

  * Loading assets
  * Physics stepping
  * Setting/resetting environment states
  * Extracting structured states for upper layers

* **task/**: Defines standardized task interfaces, enabling quick migration and extension of tasks.

* **scenario/**: Scenario configuration and management. These provide the complete set of static information required by a handler for initialization and launch, including robots, scenes, cameras, and lighting.

* **utils/**: Utility library covering cameras, kinematics, teleoperation, and more.

* **queries/**: Unified query interface for accessing simulation states and environment information.

* **example/**: Example and tutorial code.

* **tests/**: Unit and integration tests ensuring system stability.

---

### 2. **roboverse_learn - Machine Learning Module**

RoboVerse Learn focuses on implementing and integrating learning algorithms:

* **il/**: Imitation learning algorithms.
* **rl/**: Reinforcement learning algorithms.

This module provides **adapters for third-party algorithms**, as well as **in-house implementations**, and comes with **standardized training scripts**. All algorithms interact with **Metasim Tasks** via a unified interface, enabling plug-and-play learning workflows.

---

### 3. **roboverse_pack - Preconfigured Packages**

* **robots/**: Predefined robot models.
* **scenes/**: Standardized simulation scenes.
* **tasks/**: Predefined task environments.

---

### 4. **get_started - Quick Start**

* **motion_planning/**: Motion planning examples.
* **rl/**: Reinforcement learning examples.
* **dexhands/**: Dexterous hand control examples.

---

### 5. **scripts - Utility Scripts**

* **conversion/**: File and format conversion utilities.
* **advanced/**: Advanced features and extension scripts.
* **statistics/**: Data collection and statistical analysis tools.
* **mesh_tools/**: 3D mesh processing tools.

---

## Summary

The RoboVerse architecture emphasizes modularity and decoupling:

* **Metasim**: Unified simulation interface and scenario management.
* **RoboVerse Learn**: Algorithm adapters, implementations, and standardized training scripts.
* **RoboVerse Pack**: Out-of-the-box robot, scene, and task configurations.
* **Get Started**: Quick-start examples for fast adoption.
* **Scripts**: Supporting utilities and advanced tools.
