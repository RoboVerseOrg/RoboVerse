# MetaSim User Guide

**MetaSim** is the core simulation framework of RoboVerse, providing a unified interface for robotic simulation across multiple physics engines.

---

## Overview

MetaSim enables you to:

- **Write Once, Run Anywhere**: Develop simulation code that works across multiple simulators
- **Configure Declaratively**: Define robots, scenes, cameras, and physics parameters through a configuration system
- **Scale**: Run parallel environments
- **Extend**: Add new robots, tasks, and simulators through defined interfaces

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Handler** | Simulator-specific backend that manages physics and rendering |
| **Scenario** | Complete scene specification (robots, objects, cameras, lights) |
| **Task** | Gym-style environment with reward functions and success criteria |
| **State** | Unified representation of simulation state across all backends |
| **Config** | Type-safe configuration classes for all components |

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      Your Training Code                       │
├──────────────────────────────────────────────────────────────┤
│                    Task Environment (Gym API)                 │
│              reset() / step() / render() / close()            │
├──────────────────────────────────────────────────────────────┤
│                    Domain Randomization                       │
│         Physics / Visual / Sensor / Action Noise              │
├──────────────────────────────────────────────────────────────┤
│                      Scenario Config                          │
│            Robots + Objects + Cameras + Lights                │
├──────────────────────────────────────────────────────────────┤
│                      Simulator Handler                        │
│   MuJoCo │ Isaac Sim │ SAPIEN │ PyBullet │ Genesis │ ...     │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Navigation

::::{grid} 3
:gutter: 2

:::{grid-item-card} New User
:text-align: center

Start here if you're new to MetaSim

1. [Prerequisites](get_started/prerequisite)
2. [Installation](get_started/installation)
3. [Quick Start](get_started/quick_start/index)
:::

:::{grid-item-card} Building Tasks
:text-align: center

Learn to create custom tasks

1. [Concepts](concept/architecture)
2. [Configuration](concept/config)
3. [Task System](concept/task)
:::

:::{grid-item-card} Contributing
:text-align: center

Help improve MetaSim

1. [Development Guide](developer_guide/autotest)
2. [Adding Robots](developer_guide/contributing_new_robot)
3. [Code Style](developer_guide/docstring)
:::

::::

---

## Table of Contents

```{toctree}
:caption: Installation
:maxdepth: 2
:titlesonly:

get_started/prerequisite
get_started/installation
get_started/docker
get_started/advanced_installation/index
get_started/roboverse_data
```

```{toctree}
:caption: Getting Started
:maxdepth: 2
:titlesonly:

get_started/quick_start/index
get_started/advanced/index
```

```{toctree}
:caption: Concepts
:maxdepth: 2
:titlesonly:

concept/architecture
concept/state
concept/config
concept/handler
concept/task
concept/get_extras
concept/randomization
```

```{toctree}
:caption: Features
:maxdepth: 2
:titlesonly:

features/support_matrix
features/cross_embodiment
features/cross_sim
```

```{toctree}
:caption: Development Guide
:maxdepth: 2
:titlesonly:

developer_guide/architecture_review
developer_guide/autotest
developer_guide/docstring
developer_guide/precommit_hooks
developer_guide/contributing_new_robot
developer_guide/converting_asset
developer_guide/tips/index
```

```{toctree}
:caption: Troubleshooting
:maxdepth: 2
:titlesonly:

troubleshooting/common
troubleshooting/docker
troubleshooting/isaaclab
troubleshooting/known_issues/index
```
