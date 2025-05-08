FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
ARG DOCKER_USER=user
ARG HOME=/home/${DOCKER_USER}

RUN groupadd -g $DOCKER_GID $DOCKER_USER \
    && useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER \
    && echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

## Uncomment this command to change apt source if you encouter connection issues in China mainland
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
#     sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

## Install dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ssh \
    x11-apps \
    mesa-utils \
    ninja-build \
    vulkan-tools \
    libglu1 \
    && apt clean
RUN apt install -y -o Dpkg::Options::="--force-confold" sudo

USER ${DOCKER_USER}
WORKDIR ${HOME}

## Use bash instead of sh
SHELL ["/bin/bash", "-c"]

## Install conda
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p ${HOME}/conda \
    && rm "Miniforge3-$(uname)-$(uname -m).sh"
ENV PATH=${HOME}/conda/condabin:$PATH

## Initialize all future shells
RUN mamba shell init --shell bash

## Install uv
RUN wget https://astral.sh/uv/install.sh \
    && bash install.sh \
    && rm install.sh
ENV PATH=${HOME}/.local/bin:$PATH

## Clone RoboVerse
## Option 1: Clone from github
# TODO: remove this when released
# COPY --chown=${DOCKER_USER} id_ed25519 ${HOME}/.ssh/id_ed25519
# RUN ssh-keyscan github.com >> ${HOME}/.ssh/known_hosts
# RUN git clone --depth 1 --branch metasim git@github.com:RoboVerseOrg/RoboVerse.git ${HOME}/RoboVerse
## Option 2: Copy necessary files for building conda environment
COPY --chown=${DOCKER_USER} ./metasim ${HOME}/RoboVerse/metasim
COPY --chown=${DOCKER_USER} ./third_party ${HOME}/RoboVerse/third_party
COPY --chown=${DOCKER_USER} ./pyproject.toml ${HOME}/RoboVerse/pyproject.toml

WORKDIR ${HOME}/RoboVerse

## Create conda environment
RUN mamba create -n metasim python=3.10 -y \
    && mamba clean -a -y
RUN echo "mamba activate metasim" >> ${HOME}/.bashrc

## Pip install
RUN cd ${HOME}/RoboVerse \
    && eval "$(mamba shell hook --shell bash)" \
    && mamba activate metasim \
    && uv pip install -e ".[isaaclab,mujoco,genesis,sapien3,pybullet]" \
    && uv cache clean

# Test proxy connection
# RUN wget --method=HEAD --output-document - https://www.google.com/

## Install IsaacLab v1.4.1
RUN mkdir -p ${HOME}/packages \
    && cd ${HOME}/packages \
    && eval "$(mamba shell hook --shell bash)" \
    && mamba activate metasim \
    && git clone --depth 1 --branch v1.4.1 https://github.com/isaac-sim/IsaacLab.git IsaacLab \
    && cd IsaacLab \
    && sed -i '/^EXTRAS_REQUIRE = {$/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n    "robomimic": [],\n}' source/extensions/omni.isaac.lab_tasks/setup.py \
    && ./isaaclab.sh -i \
    && pip cache purge

## Install isaacgym
RUN mamba create -n metasim_isaacgym python=3.8 -y \
    && mamba clean -a -y
RUN cd ${HOME}/packages \
    && wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
RUN find ${HOME}/packages/isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
RUN cd ${HOME}/RoboVerse \
    && eval "$(mamba shell hook --shell bash)" \
    && mamba activate metasim_isaacgym \
    && uv pip install -e ".[isaacgym]" "isaacgym @ ${HOME}/packages/isaacgym/python" \
    && uv cache clean
RUN echo 'export LD_LIBRARY_PATH=${HOME}/conda/envs/metasim_isaacgym/lib/:$LD_LIBRARY_PATH' >> ${HOME}/.bashrc
RUN mkdir -p ${HOME}/conda/envs/metasim_isaacgym/lib/python3.8/site-packages/isaacgym/_bindings/src \
    && cp -r ${HOME}/packages/isaacgym/python/isaacgym/_bindings/src/gymtorch ${HOME}/conda/envs/metasim_isaacgym/lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch

## Helpful message
RUN echo 'echo "Remember to run: xhost +local:docker on the host to enable GUI applications."' >> ${HOME}/.bashrc
