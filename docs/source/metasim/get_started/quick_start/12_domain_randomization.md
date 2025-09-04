# 12. Domain Randomization

In this tutorial, we demonstrate how to implement domain randomization in MetaSim to improve the robustness and generalization of reinforcement learning agents.

## Common Usage
We provide an api that directly sets physical attributes to implement domain randomization, see the demo below.
```bash
python get_started/12_domain_randomization.py --sim isaacsim --num_envs 2 --headless
```

You can also run in interactive mode by removing the `--headless` flag to visualize the simulation.

This script demonstrates comprehensive domain randomization techniques including:
- **Mass Randomization**: Varying object masses using different probability distributions
- **Friction Randomization**: Adjusting surface friction coefficients for more realistic physics
- **Multiple Distribution Types**: Uniform, log-uniform, and Gaussian distributions
- **Flexible Operations**: Absolute value setting, scaling, and additive modifications

## Key Features

### 1. Mass Randomization
The tutorial shows how to randomize object masses using three different approaches:

- **Uniform Distribution**: Random mass within a specified range (e.g., 0.3-0.7 kg)
- **Gaussian Distribution**: Normal distribution around a mean value with controlled variance
- **Scale Operation**: Multiply existing mass by a random factor (e.g., 0.8-1.2x)

### 2. Friction Randomization
Surface friction coefficients can be randomized to simulate different materials:

- **Log-Uniform Distribution**: Useful for friction values that span multiple orders of magnitude
- **Body-Specific Control**: Apply different friction values to specific robot body parts
- **Environment Consistency**: Maintain consistent friction across all parallel environments

### 3. Distribution Types

```python
# Uniform distribution
randomize_body_mass(env, "cube", mass_range=(0.3, 0.7), distribution="uniform")

# Gaussian distribution  
randomize_body_mass(env, "sphere", mass_range=(0.2, 0.4), distribution="gaussian")

# Log-uniform distribution
randomize_body_friction(env, "franka", friction_range=(0.5, 1.5), distribution="log_uniform")
```

### 4. Operations

```python
# Absolute value setting
randomize_body_mass(env, "cube", mass_range=(0.3, 0.7), operation="abs")

# Scaling existing values
randomize_body_mass(env, "bbq_sauce", mass_range=(0.8, 1.2), operation="scale")

# Additive modification
randomize_body_mass(env, "sphere", mass_range=(0.1, 0.3), operation="add")
```

## Implementation Details

### Helper Functions

The tutorial provides two main helper functions:

1. **`randomize_body_mass()`**: Handles mass randomization with configurable distributions and operations
2. **`randomize_body_friction()`**: Manages friction coefficient randomization

### Environment Management

- **Parallel Environments**: Supports multiple parallel environments for efficient training
- **State Persistence**: Maintains initial values for before/after comparison
- **Device Handling**: Automatically handles GPU/CPU tensor placement

### Physics Properties

The script demonstrates randomization of:
- **Robot Mass**: Individual body masses for articulated robots
- **Object Mass**: Simple primitive objects (cube, sphere)
- **Surface Friction**: Material properties affecting contact dynamics

## Expected Output

When you run the script, you'll see detailed logging including:

```
=== ISAACSIM Domain Randomization Demo ===

INITIAL VALUES (Before Randomization)
============================================================
Robot body masses (shape: torch.Size([2, 7])):
  Values: [0.73  0.73  0.73  0.73  0.73  0.73  0.73]
Cube mass: [0.1] kg
Sphere mass: [0.1] kg

DOMAIN RANDOMIZATION
============================================================
Randomizing cube mass (uniform, 0.3-0.7 kg)...
  Before: [0.1] kg
  After:  [0.45] kg

RANDOMIZATION SUMMARY
================================================================================
Object          Property   Before               After                Change     
--------------------------------------------------------------------------------
Cube            Mass       [0.1]                [0.45]              Uniform    
Sphere          Mass       [0.1]                [0.31]              Gaussian   
Robot           Friction   [0.8]                [1.12]              Log-Uniform
```
## Customization

You can easily extend this tutorial by:

1. **Adding New Distributions**: Implement custom probability distributions
2. **More Properties**: Randomize additional physics properties like restitution, damping
3. **Temporal Randomization**: Vary properties over time during simulation
4. **Correlated Randomization**: Ensure related properties change together realistically

This tutorial provides a solid foundation for implementing domain randomization in your MetaSim-based reinforcement learning projects.
