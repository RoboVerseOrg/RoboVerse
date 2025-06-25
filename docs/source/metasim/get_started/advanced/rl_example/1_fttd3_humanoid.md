# 1. FastTD3 Humanoid
[FastTD3](https://github.com/younggyoseo/FastTD3) is a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, optimized for complex humanoid control tasks.
In this tutorial, we uses [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench), which have been integrated into RoboVerse. In this example we will set up and run FastTD3 within RoboVerse on the tasks of HumanoidBench.
## Environment Setup

```bash
# Step 1: Install FastTD3-specific requirements
cd RoboVerse/get_started/rl/fast_td3
pip install -r requirements.txt

# Step 2: Install RoboVerse with MJX simulator support
cd ../../../..
pip install -e ".[mjx]"
```


## One Command to Train FastTD3, Inference and Save Video
We provide tutorials for training FastTD3, inference and saving video.

Run the following command to train a humanoid agent using FastTD3:

```bash
python RoboVerse/get_started/rl/fast_td3/1_fttd3_humanoid.py
```

This script uses the following default configuration:

* Simulator: `mjx`
* Robot: `h1`
* Task: `humanoidbench:Stand`
* Environments: 1024 parallel instances

You can modify the task, robot model, or simulator by editing the `CONFIG` dictionary at the top of the script.

FastTD3 achieves fast and stable convergence:
**H1-Stand** and **H1-Walk** tasks reach success threshold in **under 10 minutes** on a **Quadro RTX 6000**.

### You can get the video like this:
#### Stand:
<div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
    <div style="width: 48%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/rl/1_fttd3_humanoid_Stand.mp4" type="video/mp4">
        </video>
    </div>
</div>

#### Walk:
<div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
    <div style="width: 48%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/rl/1_fttd3_humanoid_Walk.mp4" type="video/mp4">
        </video>
    </div>
</div>


#### Run:
<div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 20px;">
    <div style="width: 48%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/rl/1_fttd3_humanoid_Run.mp4" type="video/mp4">
        </video>
    </div>
</div>


