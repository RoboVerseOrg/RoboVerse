#  5. Hybrid Sim
In this tutorial, we will show you how to use MetaSim to simulate a hybrid system.

## Common Usage

```bash
python get_started/5_hybrid_sim.py  --sim <simulator> --renderer <renderer>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will simulate a hybrid system and it will automatically record a video. Here we demonstrate how to use one simulator for physics simulation and another simulator for rendering.


### Examples

#### IsaacLab + Mujoco
```bash
python get_started/5_hybrid_sim.py  --sim mujoco --renderer isaaclab
```

You will get the following videos:
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 50%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/5_hybrid_sim_mujoco.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Mujoco as physics engine & IsaacLab as renderer</p>
        </div>
    </div>
</div>


This hybrid simulation approach allows us to leverage the best of both worlds - the accurate physics simulation from `Mujoco` combined with the high-quality rendering capabilities of `IsaacLab`. This powerful combination enables both efficient physics computations and visually appealing results.
