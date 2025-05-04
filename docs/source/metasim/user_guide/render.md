# Render Settings

Fixed protocol:

- **Level 0:** Whole task space, same env cfg
- **Level 1:** Randomize Texture
- **Level 2:** Randomize Camera Pose
- **Level 3:** Rondomize Material/Lighting

Note that each level includes randomizations of the previous levels.

For overall Sim2Real, Randomization is applied on:

- Task space
- Material
- Texture
- Lighting
- Camera pose

Setup: Env Wrapper + Asset set
