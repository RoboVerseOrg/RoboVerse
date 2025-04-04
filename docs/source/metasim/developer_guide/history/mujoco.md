# MuJoCo Installation

This instruction is tested on Mac and Ubuntu 22.04LTS.

Create a new conda environment
```bash
conda create -n mujoco python=3.10 -y
conda activate mujoco
```

Install MuJoCo
```bash
pip install mujoco
pip install mujoco-python-viewer
pip install dm_control
pip install urdf2mjcf
```

Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub dill
```


RL dependencies

```bash
pip install hydra-core --upgrade
pip install termcolor
pip install tensorboardx
pip install gym
```

## Troubleshooting

- If you encounter the following error:

    ```
    libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: iris
    libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: iris
    libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: swrast
    ~/miniconda3/envs/mujoco/lib/python3.10/site-packages/glfw/__init__.py:917: GLFWError: (65543) b'GLX: Failed to create context: BadValue (integer parameter out of range for operation)'
    warnings.warn(message, GLFWError)
    ~/miniconda3/envs/mujoco/lib/python3.10/site-packages/glfw/__init__.py:917: GLFWError: (65538) b'Cannot set swap interval without a current OpenGL or OpenGL ES context'
    warnings.warn(message, GLFWError)
    python: /builds/florianrhiem/pyGLFW/glfw-3.4/src/window.c:711: glfwGetFramebufferSize: Assertion `window != NULL' failed.
    [1]    69573 IOT instruction (core dumped)
    ```
    Try
    ```bash
    conda install -c conda-forge libstdcxx-ng  # see https://stackoverflow.com/a/71421355
    ```
