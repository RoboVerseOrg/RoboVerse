# Installation with Docker

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
