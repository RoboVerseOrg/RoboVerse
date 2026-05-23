---
orphan: true
---

# Dataset and Benchmark

This section documents the RoboVerse task suite, asset library, and evaluation protocols.

---

## Quick Links

### For Researchers

- [Task Overview](tasks/overview.md) - Browse available tasks by category
- [Benchmark Usage](benchmark/usage.md) - Run standardized evaluations
- [Benchmark Results](benchmark/results.md) - Compare with state-of-the-art

### For Developers

- [Robot Configurations](dataset/robots.md) - Available robot embodiments
- [Object Assets](dataset/objects.md) - Object models and properties
- [Scene Definitions](dataset/scenes.md) - Pre-built simulation scenes

---

## Task Categories

### Manipulation Tasks

Single-arm and bimanual manipulation tasks including:

- **Pick & Place**: Grasping and relocating objects
- **Insertion**: Peg-in-hole and assembly tasks
- **Tool Use**: Using tools to manipulate objects
- **Articulated Objects**: Doors, drawers, buttons, switches

### Locomotion Tasks

Whole-body control and locomotion tasks:

- **Walking**: Forward, backward, turning gaits
- **Running**: High-speed locomotion
- **Terrain Navigation**: Stairs, slopes, rough terrain
- **Recovery**: Fall recovery and balance

### Mobile Manipulation

Combined navigation and manipulation:

- **Fetch Tasks**: Navigate to object, pick, deliver
- **Household Tasks**: Cleaning, organizing, cooking assistance

---

## Benchmark Suite

The RoboVerse Benchmark provides standardized evaluation protocols for:

| Benchmark | Tasks | Metrics | Purpose |
|-----------|-------|---------|---------|
| **RV-Manip** | 50 | Success Rate, SPL | Manipulation generalization |
| **RV-Loco** | 20 | Distance, Energy | Locomotion efficiency |
| **RV-Cross** | 30 | Transfer Success | Cross-embodiment transfer |

See [Benchmark Overview](benchmark/overview.md) for detailed evaluation protocols.

