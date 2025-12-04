#!/bin/bash

# Install additional dependencies
echo "Install additional dependencies..."
# Fix .zarr issue
pip install zarr==2.16.1 blosc==1.11.1
pip install numcodecs==0.11.0

# Fix hydra issue
pip install --upgrade hydra-core

