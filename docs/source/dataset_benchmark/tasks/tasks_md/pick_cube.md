# pick_cube

![dense-reward][dense-reward-badge]
![sparse-reward][sparse-reward-badge]
![demos][demos-badge]

[dense-reward-badge]: https://img.shields.io/badge/dense-yes-brightgreen.svg
[sparse-reward-badge]: https://img.shields.io/badge/sparse-yes-brightgreen.svg
[demos-badge]: https://img.shields.io/badge/demos-yes-brightgreen.svg

**[🔗 Official Task Page](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#stackcube-v1)**


**Task Description:** A simple task where the objective is to grasp a red cube with the Panda robot and move it to a target goal position. This is also the baseline task to test whether a robot with manipulation capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube’s xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube’s z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within goal_thresh (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)

<div style="display: flex; justify-content: center;">
  <video preload="none" controls width="100%" style="max-width: min(100%, 512px);">
    <source src="pick_cube.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
