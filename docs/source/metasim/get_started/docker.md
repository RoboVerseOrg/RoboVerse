# 🐳 Docker

## Prerequisites

Please make sure you have installed `docker` in the officially recommended way. Otherwise, please refer to the [official guide](https://docs.docker.com/engine/install/ubuntu/).

Please install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) following the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), or run the following commands:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Please create and add the docker user information to `.env` file. To use the same user information as the host machine, run in project root:
```bash
printf "DOCKER_UID=$(id -u $USER)\nDOCKER_GID=$(id -g $USER)\nDOCKER_USER=$USER\n" > .env
```

## Build the docker image

Build the docker image and attach to the container bash:
```bash
docker compose up --build -d && docker exec -it metasim bash
```
This will automatically build docker image `roboverse-metasim`.

It may take ~10mins when the network speed is ~25MB/s. The docker image size would be 35~40GB.

## Run the docker container in VSCode/Cursor

Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VSCode/Cursor.

Then reopen the window, click the `Reopen in Container` option in the bottom left corner.

## Setup GUI

Before you run any command, you need to setup the GUI. On the host machine, run:
```bash
xhost +local:docker
```

In container, launch a xclock application to test the GUI:
```bash
xclock
```

If a clock successfully shown on the host machine, the GUI is working.

## Tips

### Run docker without sudo

You may want to run docker without sudo. Run:
```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
```
After re-login, you should be able to run docker without sudo:
```bash
docker run hello-world
```

### Setup proxy for docker

1. Set up local Clash proxy and make sure it works on local IP address. For example, you need enable "Allow LAN" if you are using Clash.

    Turn on clash to allow LAN:

    ```
    # vim ~/Clash/config.yaml
    allow-lan: true
    ```

    Then test in your terminal

    ```
    export HOST_IP=192.168.61.221
    export all_proxy=socks5://${HOST_IP}:7890
    export all_proxy=socks5://${HOST_IP}:7890
    export https_proxy=http://${HOST_IP}:7890
    export http_proxy=http://${HOST_IP}:7890
    export no_proxy=localhost,${HOST_IP}/8,::1
    export ftp_proxy=http://${HOST_IP}:7890/

    # check env variables are set
    env | grep proxy

    # test connection
    curl -I https://www.google.com
    ```

2. Set up docker proxy.
    ```
    # vim ~/.docker/config.json
    "proxies": {
        "default": {
            "httpProxy": "http://192.168.1.55:7890",
            "httpsProxy": "http://192.168.1.55:7890",
            "allProxy": "socks5://192.168.1.55:7890",
            "noProxy": "192.168.1.55/8"
        }
    }
    ```
    ```{note}
    Do NOT set IP address to `127.0.0.1`. Instead, change it to your local ipv4 address.
    ```

3. Setup proxy mirros used when docker pull, etc

    ```
    # sudo vim /etc/docker/daemon.json
    {
        ...
        "registry-mirrors": [
            "https://mirror.ccs.tencentyun.com",
            "https://05f073ad3c0010ea0f4bc00b7105ec20.mirror.swr.myhuaweicloud.com",
            "https://registry.docker-cn.com",
            "http://hub-mirror.c.163.com",
            "http://f1361db2.m.daocloud.io"
        ]
    }
    ```

4. Restart docker [and then build again]
    ```
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

5. Add PROXY to `.env` file.
    ```
    DOCKER_USER=...
    DOCKER_UID=...
    DOCKER_GID=...
    PROXY=http://192.168.1.55:7890
    ```

6. Uncomment the lines in dockerfile which changes ubuntu apt sources to aliyun if you encounter `apt install` failures.
    ```
    # Change apt source if you encouter connection issues
    RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
        sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
    ```

7. Be patient. Sometimes you need run `docker compose build` multiple times.
   
### Setup docker for NVIDIA RTX50 series GPUs

For RTX50 series GPUs, the following environments are required. 
| Component      | Version   | Notes                          |
|----------------|-------------------|--------------------------------|
| 🐧 OS           | Ubuntu ≥ 22.04     | Required by IsaacLab        |
| 🐍 Python       | python == 3.10     | Required by multiple simulators        |
| 🔥 PyTorch      | torch ≥ 2.7.1      | Required by RTX50 series GPUs        |
| 🚀 CUDA         | CUDA ≥ 12.8        | Required by RTX50 series GPUs |
```{note}
Currently, the IsaacGym does not support the NVIDIA RTX50 series GPUs, as it is limited to `python==3.8` or earlier.
```
1. Pull the official NVIDIA image.

    To make sure the docker environment supports RTX50 series GPUs and cuda 12.8. Please pull the official Ubuntu 22.04 base image that supports cuda 12.8 from NVIDIA by running the following commands:

    ```bash
    docker pull nvidia/cuda:12.8.0-base-ubuntu22.04
    ```
2. Setup docker environments.
   
    Please run the base image with GPU supporting and install necessary development tools (build-essential, CMake, git, etc.).

    ```bash
    docker run --gpus all -it nvidia/cuda:12.8.0-base-ubuntu22.04

    apt-get update && apt-get install -y --no-install-recommends build-essential cmake git curl wget ca-certificates pkg-config software-properties-common unzip nano sudo
    ```

    Then, setup the conda environment with `python==3.10` for RoboVerse:
    ```bash
    conda create -n roboverse python=3.10
    ```
3. Setup RoboVerse-IsaacLab environments.

    Please pull the RoboVerse official code repository:
    ```bash
    git clone https://github.com/RoboVerseOrg/RoboVerse.git

    cd RoboVerse
    ```

    The environment in the `pyproject.toml` is currently not compatible for NVIDIA RTX50 series GPUs. Please use `pip` to install isaacsim manually.

    ```bash
    pip install protobuf
    pip install pyglet
    pip install isaacsim==4.2.0.2
    pip install isaacsim-extscache-physics==4.2.0.2
    pip install isaacsim-extscache-kit==4.2.0.2
    pip install isaacsim-extscache-kit-sdk==4.2.0.2
    ```

    Please install the IsaacLab dependencies by running following commands:

    ```bash
    cd third_party

    wget https://codeload.github.com/isaac-sim/IsaacLab/zip/refs/tags/v1.4.1 -O IsaacLab-1.4.1.zip && unzip IsaacLab-1.4.1.zip

    cd IsaacLab-1.4.1

    sed -i '/^EXTRAS_REQUIRE = {$/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n    "robomimic": [],\n}' source/extensions/omni.isaac.lab_tasks/setup.py

    ./isaaclab.sh -i
    ```

    After installing the IsaacLabv 1.4, the torch will be modified to 2.4.0, reinstall the torch to 2.7.1. The `torch==2.4.0` will not be compatible with NVIDIA RTX50 series GPUs.
    ```bash
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

    Finally, please install the necessary libraries required by IsaacLab.
    ```bash
    pip install rootutils
    pip install tyro
    pip install loguru
    pip install open3d
    ```
4. Setup RoboVerse-Mujoco environments.

    After setting up issaclab, mujoco can be easily installed with the following command:
    ```bash
    pip install mujoco 
    pip install dm-control
    ```
5. Setup RoboVerse-Reinforcement Learning environments.

    RoboVerse provides two reinforcement learning demos: [PPO Reaching](https://roboverse.wiki/metasim/get_started/advanced/rl_example/0_ppo_reaching#ppo-reaching) and [FastTD3 Humanoid](https://roboverse.wiki/metasim/get_started/advanced/rl_example/1_fttd3_humanoid). To run these two demos, please follow the steps below to setup your environments.

    Setup the PPO environments.
    ```bash
    pip install stable-baselines3
    ```

    Setup the FastTD3 environments.
    ```bash
    pip install mujoco-mjx
    pip install dm-control
    pip install jax[cuda12]
    pip install wandb
    pip install tensordict
    ```

## ⚠️ Trouble Shooting
The following issues may arise in various modules during Docker configuration. 	Corresponding solutions are provided for each case.

### RoboVerse-IsaacLab
### 1. "\[omni.gpu_foudation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode."
This problem is due to incorrect startup method of docker images and the docker cannot access the phsical GPUs in the host. 

Save the running docker container to the docker images.
```bash
docker ps -a

docker commit your_container_id your_image_name
```
Rerun the image by following setting, which allows the docker to call the physical GPUs in the host.
```bash
docker run -it --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/dri:/dev/dri \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
-v /usr/share/vulkan/implicit_layer.d:/usr/share/vulkan/implicit_layer.d:ro \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e XDG_RUNTIME_DIR=/run/user/$(id -u) \
your_image_name /bin/bash
```

### 2. Errors about shared libraries not found. For example, "OSError: libGL.so.1: cannot open shared object file: No such file or directory."
This issue arises from the absence of required dynamic libraries.

Please install the necessary dynamic libraries for IsaacLab.
```bash
sudo apt-get install libgl1-mesa-glx libsm6 libxt6 libglu1-mesa
```

### 3. Errors related to numpy version. "Error: partially initialized module 'numpy.core.arrayprint' has no attribute 'array2string'."
The RoboVerse-IsaacLab is sensitive to numpy version, currently it only supports for `numpy < 2`. Please install the `numpy < 2`.
```bash
pip install numpy==1.26.4
```

### 4. "TypeError: array.\_\_init\_\_() got an unexpected keyword argument 'owner'."
This seems a bug in the omni library function.

The corresponding file is:
```
/home/anaconda3/envs/roboverse/lib/python3.10/site-packages/isacsim/extscache/omni.replicator.core-1.11.20+186.1.8.1x64.r.cp310/omni/replicatorcore/scripts/utils/annotator_utils.py
```
Delete the `owner` parameter in corresponding function (line 341) in the `annotator_utils.py` file.


### RoboVerse-Mujoco
### 1. "ValueError: Image width 1024 > framebuffer width 640. Either reduce the image width or specify a larger offscreen framebuffer in the model XML using clause."
This is a bug from the new version of RoboVerse (After the `/RoboVerse/metasim/sim/mujoco/mujoco.py` updating from 2025.07.11).

A temporary solution is to hard code the `camera.width=640` and `camera.height=480` in `/RoboVerse/metasim/sim/mujoco/mujoco.py`
lines 341.

### RoboVerse-Reinforcement Learning
### 1. "ModuleNotFoundError: No mudule named 'metasim'"
This issue is due to the project path is not correctly setting.

Add project path at the beginning of `/RoboVerse/get_started/rl/fast_td3/1_fttd3_humanoid.py`
```python
roboverse_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

sys.path.append(roboverse_path)
```

### 2. "ImportError: Cannot initialize a headless EGL display."
This issue is due to that the docker could not find the `EGL` engine for rendering.

You can manually set environment variables in docker:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Otherwise, you can add these environment variables in the startup command of the docker image:
```bash
docker run -it --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/dri:/dev/dri \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
-v /usr/share/vulkan/implicit_layer.d:/usr/share/vulkan/implicit_layer.d:ro \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e XDG_RUNTIME_DIR=/run/user/$(id -u) \
-e MUJOCO_GL=egl \
-e PYOPENGL_PLATFORM=egl \
-e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
your_image_name /bin/bash
```
### 3. "AttributeError: 'NoneType' object has no attribute 'eglQueryString'"
This issue is due to the lack of necessary dynamic libraries for EGL engine. Please install them via the following commands:
```bash
sudo apt-get install libegl1 libgl1-mesa-glx
```

### 4. "jaxlib.\_jax.XlaRuntimeError: FAILED\_PRECONDITION: DNN library initialization failed"
The FastTD3 requires `cudnn >= 9.8.0`. Please install new version of cudnn.
```bash
pip install nvidia-cudnn-cu12==9.10.2.21
```
