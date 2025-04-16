// Author: chenxi-wang

#include "cylinder_query.h"
#include "utils.h"

void query_cylinder_point_kernel_wrapper(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx);

at::Tensor cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, const float radius, const float hmin, const float hmax,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(rot);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(rot);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(rot);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_cylinder_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, hmin, hmax, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), rot.data<float>(), idx.data<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return idx;
}
