# Common Troubleshooting

## libGL error
If you encounter the following error:
```
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: swrast
```
Try
```bash
conda install -c conda-forge libstdcxx-ng
```
For more details, see [this answer](https://stackoverflow.com/a/71421355).
