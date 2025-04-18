# IsaacLab Troubleshooting

## CPU usage
Sometimes, IsaacLab uses all the CPU cores and crash the machine. If that happens, you can limit the CPU cores in use by `taskset`. For example, if you want to use 0,1,2,3 cores, you can run:
```bash
taskset -c 0,1,2,3 python ...  # your script
```

## GLXBadFBConfig
If you meet the following error:
```
X Error of failed request:  GLXBadFBConfig
  Major opcode of failed request:  150 (GLX)
  Minor opcode of failed request:  0 ()
  Serial number of failed request:  136
  Current serial number in output stream:  136
```
You can try to run:
```bash
export MESA_GL_VERSION_OVERRIDE=4.6
```
Or
```bash
export MESA_GL_VERSION_OVERRIDE=4.5
```

## Occupied GPU memory
If you interrupt the process, the GPU memory may not be released because **the forked process is not killed**. You need to manually kill it by running:
```bash
pkill -9 pt_main_thread && pkill -9 python
```

```{note}
This will risk killing other python processes.
```

## No Space left on device
If you meet `Failed to create change watch for ... No space left on device`, it doesn't really matter. You can simply ignore it, or change your system setting by running `echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf` or just simply restart the system.

## libGLU problem
If you are running on a server in headless mode, and meet `Failed to open /.../iray/libneuray.so: libGLU.so.1: cannot open shared object file: No such file or directory`, try:
```bash
sudo apt-get update
sudo apt-get install libglu1-mesa
```

## Bad rendering quality
If you get the rendered image with severe aliasing artifacts, it could be very tricky bugs either in IsaacLab, IsaacSim or CUDA. We saw this bug in some machines, but couldn't reproduce it on other machines, and find no way to debug. If you unfortunatedly get into this bug, we recommand you to use other machines and try again. Or, using docker may help.

## Stuck at `OmniGraphSettings::getCudaDeviceOrdinal: unable to get a valid CUDA device id from the renderer.`
Please refer to [this issue](https://github.com/isaac-sim/IsaacLab/issues/987).
