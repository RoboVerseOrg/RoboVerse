# Common Troubleshooting

## libGL error
Error message:
```
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: swrast
```

Please install the necessary dynamic libraries for rendering.
```bash
conda install -c conda-forge libstdcxx-ng
```
For more details, see [this answer](https://stackoverflow.com/a/71421355).

## MuJoCo: egl error
Error message:
```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'"
```

This issue is due to the lack of necessary dynamic libraries for EGL engine. Please install them via the following commands:
```bash
sudo apt-get install libegl1 libgl1-mesa-glx
```

## MJX: DNN library initialization failed
Error message:
```
jaxlib.\_jax.XlaRuntimeError: FAILED\_PRECONDITION: DNN library initialization failed"
```
The FastTD3 requires `cudnn >= 9.8.0`. Please install new version of cudnn.
```bash
pip install nvidia-cudnn-cu12==9.10.2.21
```
