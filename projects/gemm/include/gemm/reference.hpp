#pragma once

namespace gemm {

void reference_cpu(const float* a, const float* b, float* c, int m, int n, int k);

}  // namespace gemm
