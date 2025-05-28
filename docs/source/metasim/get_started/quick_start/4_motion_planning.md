# 4. Motion Planning
In this tutorial, we will show you how to use MetaSim to plan a motion for a robot.
Note here, we use the `curobo` package to plan the motion. If you haven't installed it, please refer to our [curobo installation guide](http://127.0.0.1:8000/metasim/get_started/advanced_installation/curobo.html).

## Common Usage

```bash
python get_started/4_motion_planning.py  --sim <simulator>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

By running the above command, you will plan a motion for a robot and it will automatically record a video.


### Examples

#### IsaacLab
```bash
python get_started/4_motion_planning.py  --sim isaaclab
```

#### Isaac Gym
```bash
python get_started/4_motion_planning.py  --sim isaacgym
```

#### Mujoco
```bash
python get_started/4_motion_planning.py  --sim mujoco --headless
```
Note that we find the `non-headless` mode of Mujoco is not stable. So we recommend using the `headless` mode.


#### Genesis
```bash
python get_started/4_motion_planning.py  --sim genesis
```
Note that we find the `headless` mode of Genesis is not stable. So we recommend using the `non-headless` mode.

#### Sapien
```bash
python get_started/4_motion_planning.py  --sim sapien3
```

#### Pybullet
```bash
python get_started/4_motion_planning.py  --sim pybullet
```

You will get the following videos:

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">IsaacLab</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_mujoco.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Mujoco</p>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_genesis.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Genesis</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_sapien3.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Sapien</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/4_motion_planning_pybullet.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">PyBullet</p>
        </div>
    </div>

</div>
