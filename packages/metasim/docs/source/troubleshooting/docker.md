# Docker Troubleshooting

## MuJoCo: Cannot initialize a headless EGL display
Error message:
```
ImportError: Cannot initialize a headless EGL display."
This issue is due to that the docker could not find the `EGL` engine for rendering.
```

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

## IsaacLab: Failed to create any GPU devices
Error message:
```
[omni.gpu_foudation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode.
```

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
