# Tutorial 5: Hybrid Simulation

**Objective**: Learn how to use different simulators for physics and rendering to get the best of both worlds.

**What you'll learn**:
- Separating physics simulation from rendering
- Using MuJoCo for fast physics with Isaac Sim for high-quality rendering
- When to use hybrid simulation

**Prerequisites**: Completed [Tutorial 4: Motion Planning](4_motion_planning)

**Estimated time**: 25 minutes

---

Hybrid simulation allows you to combine a fast physics engine with a high-quality renderer. This is useful when you need both computational efficiency and photorealistic images.

## Running the Tutorial

```bash
python get_started/5_hybrid_sim.py  --sim <simulator> --renderer <renderer>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will simulate a hybrid system and it will automatically record a video. Here we demonstrate how to use one simulator for physics simulation and another simulator for rendering.


### Examples

#### IsaacSim + Mujoco
```bash
python get_started/5_hybrid_sim.py  --sim mujoco --renderer isaacsim
```

You will get the following videos:
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 50%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/5_hybrid_sim_mujoco.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Mujoco as physics engine & IsaacSim as renderer</p>
        </div>
    </div>
</div>


This hybrid simulation approach allows us to leverage the best of both worlds - the accurate physics simulation from `Mujoco` combined with the high-quality rendering capabilities of `IsaacSim`. This powerful combination enables both efficient physics computations and visually appealing results.
