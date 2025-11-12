#!/bin/bash

# Install diffusion_policy
echo "Install diffusion_policy..."
cd ./roboverse_learn/il/utils/diffusion_policy || { echo "diffusion_policy do not exit"; exit 1; }
pip install -e .

# Install act
echo "Install act..."
cd ../../../../
cd roboverse_learn/il/utils/act/detr || { echo "detr do not exit"; exit 1; }
pip install -e .

# Install additional dependencies
echo "Install additional dependencies..."
cd ../../../../../
pip install pandas wandb

# Fix .zarr issue
pip install zarr==2.16.1 blosc==1.11.1
pip install numcodecs==0.11.0

# Fix hydra issue
pip install --upgrade hydra-core

# dp-VITA additional dependency
pip install torchcfm
