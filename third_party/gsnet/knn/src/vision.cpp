#include "knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn, "k-nearest neighbors");
}
