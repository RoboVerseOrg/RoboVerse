# 0. Franka Planning
## Franka Planning Understanding

Motion Planning is always challenging and complex to tune. Every time you want to plan a motion, you need to tune the rotation convention, the planning library, etc. We provide a simple tutorial to show you how to plan a motion for the Franka robot.

First, you need to install [curobo](https://roboverse.wiki/metasim/get_started/advanced_installation/curobo). RoboVerse provides a clean readme to help you install the dependencies. Follow the link to install them!


Then run:
```bash
python get_started/motion_planning/0_franka_planning.py --sim <simulator>
```

Here is a visualization of how we plan the motion for the Franka robot:
<div style="text-align: center;">
    <img src="../../../_static/standard_output/motion_planning/franka_planning_understanding.png" width="40%"/>
</div>
<br>

You will get the following videos:

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;">
    <div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/motion_planning/0_franka_planning_isaaclab.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Lab</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/motion_planning/0_franka_planning_isaacgym.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Isaac Gym</p>
        </div>
        <div style="width: 32%; text-align: center;">
            <video width="100%" autoplay loop muted playsinline>
                <source src="https://roboverse.wiki/_static/standard_output/motion_planning/0_franka_planning_genesis.mp4" type="video/mp4">
            </video>
            <p style="margin-top: 5px;">Genesis</p>
        </div>
    </div>

</div>
