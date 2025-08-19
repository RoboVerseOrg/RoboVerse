// Author: chenxi-wang

#pragma once
#include <torch/extension.h>

at::Tensor cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, const float radius, const float hmin, const float hmax,
                      const int nsample);
