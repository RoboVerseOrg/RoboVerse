# Pico teleop installation
## install XRoboToolkit-PC-Service
```bash
sudo dpkg -i roboticsservice_1.0.0.0_amd64.deb
```

## Build python binding
```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind
./setup_ubuntu.sh
rm -rf XRoboToolkit-PC-Service/
```
