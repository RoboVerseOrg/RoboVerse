# Domain Randomization

This example demonstrates comprehensive domain randomization using all five randomizers available in RoboVerse. The demo implements a progressive randomization strategy with five levels of increasing complexity.

## Overview

Domain randomization is a technique for training robust policies by varying simulation parameters. This example shows how to use RoboVerse's randomization framework to systematically vary object properties, visual appearance, camera viewpoints, and scene geometry.

## Files

- `12_domain_randomization.py` - Main demonstration script
- `README_DOMAIN_RANDOMIZATION.md` - This documentation

## Randomization Levels

The demo implements a progressive randomization strategy inspired by the RoboVerse paper:

| Level | Description | Active Randomizers | Object Placement |
|-------|-------------|-------------------|------------------|
| 0 | Baseline (no randomization) | None | Ground |
| 1 | Object properties | ObjectRandomizer | Ground |
| 2 | Visual appearance | ObjectRandomizer, MaterialRandomizer, LightRandomizer | Ground |
| 3 | Camera viewpoint | All above + CameraRandomizer | Ground |
| 4 | Full scene geometry | All above + SceneRandomizer | On table surface |

## Usage

### Basic Usage

```bash
# Run with different randomization levels
python get_started/12_domain_randomization.py --level 0
python get_started/12_domain_randomization.py --level 1
python get_started/12_domain_randomization.py --level 2
python get_started/12_domain_randomization.py --level 3
python get_started/12_domain_randomization.py --level 4
```

### Advanced Options

```bash
# Specify random seed for reproducibility
python get_started/12_domain_randomization.py --level 2 --seed 42

# Change simulation parameters
python get_started/12_domain_randomization.py --level 2 --num-steps 200 --randomize-interval 5

# Run headless (no GUI)
python get_started/12_domain_randomization.py --level 2 --headless
```

## Scene Configuration

### Objects

The demo includes three objects with different randomization strategies:

- **cube**: Manipulable wooden block with full randomization (physics properties, pose, and material)
- **sphere**: Bouncy rubber ball with full randomization (optimized for high restitution)
- **box_base**: Fixed articulated object serving as environmental prop (material randomization only)
- **franka**: Panda robot arm (static, for context)

### Level 0-3 Configuration

Objects are placed on the ground plane:
- Open space with no walls or ceiling
- 2 light sources providing 1500 cd total illumination
- Camera positioned at (1.5, -1.5, 1.5) looking down at origin

### Level 4 Configuration

Level 4 introduces a complete tabletop workspace:

**Environment**:
- Enclosed room: 10m x 10m x 5m
- Table surface: 1.8m x 1.8m at 0.7m height
- All geometry includes physics collision

**Lighting**:
- 4 light sources with 29000 cd total intensity
- All lights configured as global to penetrate walls
- Fixed positions inside room to avoid occlusion

**Object Placement**:
- Objects placed on table surface (z = 0.7-0.8m)
- Robot base positioned on table (z = 0.7m)
- Physics simulation ensures objects remain on table

## Randomizers

### 1. ObjectRandomizer (Level 1+)

Randomizes physical properties and pose:
- Mass variation for different object weights
- Friction coefficient for contact dynamics
- Position and orientation for varied initial states

**Presets used**:
- `grasping_target`: For cube (varied grasping challenges)
- `bouncy_object`: For sphere (high restitution for bouncing)

### 2. MaterialRandomizer (Level 2+)

Randomizes visual appearance and material physics:
- MDL textures for realistic rendering (wood materials)
- PBR materials for efficient rendering (rubber)
- Physical properties (friction, restitution)

**Materials**:
- Cube: Wood (MDL texture library, ~50 variants)
- Sphere: Rubber (PBR with high restitution)
- Box: Wood (MDL texture library)

### 3. LightRandomizer (Level 2+)

Randomizes lighting conditions:
- Intensity variation (brightness changes)
- Color temperature shifts
- Position changes (Level 0-3 only)

**Configuration**:
- Level 0-3: Combined randomization (intensity, color, position)
- Level 4: Intensity-only (positions fixed to avoid wall occlusion)

### 4. CameraRandomizer (Level 3+)

Randomizes camera viewpoint:
- Small position adjustments (micro-variations)
- Orientation changes
- Intrinsic parameters (focal length, FOV)

Uses surveillance camera preset with micro-adjustment strategy.

### 5. SceneRandomizer (Level 4)

Creates and randomizes scene geometry:
- Floor with material variation (~150 materials)
- Four walls forming enclosure
- Ceiling
- Table surface with material variation (~300 materials)

All scene geometry includes static collision for physics interaction.

## Implementation Details

### Level 4 Setup Sequence

The setup sequence for Level 4 is carefully orchestrated to ensure proper physics:

1. Create scenario configuration
2. Initialize simulation handler
3. Create scene geometry (table, walls, floor, ceiling) via SceneRandomizer
4. Allow scene to stabilize (10 simulation steps)
5. Place objects on table surface
6. Stabilize physics (20 simulation steps)
7. Verify object positions
8. Begin randomization loop

This sequence ensures that the table exists before objects are placed, preventing objects from falling through.

### Lighting Strategy

**Open Space (Level 0-3)**:
- Minimal lighting sufficient for unobstructed view
- 2 light sources: 1 distant (1000 cd) + 1 sphere (500 cd)
- Lights can randomize freely without occlusion concerns

**Enclosed Room (Level 4)**:
- Significantly increased lighting to compensate for wall absorption
- 4 light sources with 29000 cd total
- All lights configured as global to penetrate geometry
- Light positions fixed during randomization to maintain consistent illumination

### Physics Collision

Scene geometry uses CollisionAPI for static collision:
- Floor, walls, ceiling, and table all have collision enabled
- Static geometry (mass = 0 implicitly) does not fall or move
- Provides support surface for dynamic objects

Note: Using RigidBodyAPI would make geometry dynamic and subject to gravity. For static scene elements, only CollisionAPI is needed.

### Material Randomization

The demo uses two material systems:
- **MDL materials**: High-quality NVIDIA materials requiring material files
- **PBR materials**: Procedural materials using metallic-roughness workflow

Materials are sampled from curated collections:
- Wood: ~50 variants (ARNOLD library)
- Metal: ~30 variants (vMaterials library)
- Floor: ~150 variants (carpet, wood, stone)
- Table: ~300 variants (wood, stone, metal)

## Output

Videos are saved to `get_started/output/`:
- `12_dr_level0_isaacsim.mp4` - Baseline
- `12_dr_level1_isaacsim.mp4` - Object randomization
- `12_dr_level2_isaacsim.mp4` - Visual randomization
- `12_dr_level3_isaacsim.mp4` - Camera randomization
- `12_dr_level4_isaacsim.mp4` - Full domain randomization

## Requirements

- IsaacSim installation
- Material libraries in `roboverse_data/materials/`
- Asset files in `roboverse_data/assets/`
