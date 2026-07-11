#pragma once

#include <cuda_runtime_api.h>

namespace gemm {

struct Problem;

void reference_cpu(const float* a, const float* b, float* c, int m, int n, int k);
void reference_cublas_device(const float* device_a, const float* device_b,
                             float* device_c, Problem problem,
                             cudaStream_t stream);

}  // namespace gemm
