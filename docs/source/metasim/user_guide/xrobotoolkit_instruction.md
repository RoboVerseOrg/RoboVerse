# XR Teleoperation Setup

Follow these steps to install and run a full XR-to-robot teleoperation sample on a **PICO 4 Ultra headset and a Linux x86 PC**.

- Linux x86 PC: Ubuntu 22.04

- PICO 4 Ultra: User OS >5.12. Currently supports [PICO 4 Ultra](https://www.picoxr.com/global/products/pico4-ultra) and [PICO 4 Ultra Enterprise](https://www.picoxr.com/global/products/pico4-ultra-enterprise).

1. **Install Roboverse**
   - Install Roboverse with your preferred simulators following [this instruction](https://roboverse.wiki/metasim/get_started/installation), the XR teleoperation is tested with Isaac v1.4
   - Install [cuRobo](https://roboverse.wiki/metasim/get_started/advanced_installation/curobo)

2. **Install xrobotoolkit-sdk**
   - Use the following command
        ```bash
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
        cd XRoboToolkit-PC-Service-Pybind
        ./setup_ubuntu.sh
        ```

3. **Install XRoboToolkit-PC-Service**
   - Download [deb package for ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb), or build from the [repo source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).
   - To install, use command
     ```bash
      sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
      ```

4. **Install the XR App on Headset**
   - Turn on developer mode on Pico 4 Ultra headset first ([Enable developer mode on Pico 4 Ultra](https://developer.picoxr.com/ja/document/unreal/test-and-build/)), and make sure that [adb](https://developer.android.com/tools/adb) is installed properly.
   - Download [XRoboToolkit-PICO.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.0.1/XRoboToolkit-PICO-1.0.1.apk) on a PC with adb installed.
   - To install apk on the headset, use command
     ```bash
      adb install -g XRoboToolkit-PICO.apk
      ```
5. **Run XR teleoperation in Roboverse**
   - Connect robot PC and Pico 4 Ultra under the same network
   - On robot PC, double click app icon of `XRoboToolkit-PC-Service` or run service `/opt/apps/roboticsservice/runService.sh`
   - Open app `XRoboToolkit` on the Pico headset. Details of the Unity app can be found in the [Unity source repo](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client).
   - Run the XR teleoperation demo
    ```bash
    python metasim/scripts/teleop_xr.py --task=PickCube
    ```
