# stack_cube

![dense-reward][dense-reward-badge]
![sparse-reward][sparse-reward-badge]
![demos][demos-badge]

[dense-reward-badge]: https://img.shields.io/badge/dense-yes-brightgreen.svg
[sparse-reward-badge]: https://img.shields.io/badge/sparse-yes-brightgreen.svg
[demos-badge]: https://img.shields.io/badge/demos-yes-brightgreen.svg

**[🔗 Official Task Page](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#stackcube-v1)**


**Task Description:** The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.

**Randomizations:**
- both cubes have their z-axis rotation randomized
- both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

**Success Conditions:**
- the red cube is on top of the green cube (to within half of the cube size)
- the red cube is static
- the red cube is not being grasped by the robot (robot must let go of the cube)

<div style="display: flex; justify-content: center;">
  <video preload="none" controls width="100%" style="max-width: min(100%, 512px);">
    <source src="stack_cube.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
