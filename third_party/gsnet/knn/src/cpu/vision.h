#pragma once
#include <torch/extension.h>

void knn_cpu(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, long* ind_buf);