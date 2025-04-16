// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)
