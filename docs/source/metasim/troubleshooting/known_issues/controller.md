# Unaligned Control

## Align the controller

It remains a problem to align the controller across different simulators. Consider the following cases:

1. For **Manipulation tasks using grippers**, the simulation gap is relatively small.
For this case, it is reasonable to use the default controller in different simulators, and the *end-effector trajectory* is
relatively consistent across different simulators.

2. For **Locomotion tasks**, the simulation gap is large, considering that Sim2Sim is even used to evaluate the policy performance.
For this case, it is not practical to align the controller across different simulators, and there is no such concept of *end-effector trajectory*.

3. For **Manipulation tasks using dexterous hands**, the simulation gap is larger than grippers. It is somewhere between the above two cases.

### Current approach
For the current stage of our project:
- **Case 1 (Grippers Manipulation)**: We ignore the controller alignment problem, as the default controllers are sufficient.
- **Case 2 (Locomotion)**: We bypass the controller alignment issue by using state replay techniques.

## End-effector trajectory

For manipulation tasks with grippers, what are we actually talking about when we say *end-effector trajectory*? There could be two cases:
1. The **target position** of the end-effector.
2. The **actual position** of the end-effector given the target position and the controller.

For most circumstances, the above two cases are aligned, meaning that when we apply a target position to the end-effector, we do want it to reach that position precisely.

However, in some cases, the above two cases are not aligned. For example, when we use a policy trained from reinforcement learning, the target position may lack constraints, leading to exaggerated trajectory that end-effector might not be able to replicate accurately.

### Current approach
For the current stage of our project, we do not consider the end-effector trajectory problem. As long as the trajectory can be replayed successfully, we keep it.

TODO: Add a table to illustrate which case is used for tasks from each source benchmark.
