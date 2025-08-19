#pragma once
#include <torch/extension.h>
// #include <THC/THC.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/ThrustAllocator.h>

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, cudaStream_t stream);