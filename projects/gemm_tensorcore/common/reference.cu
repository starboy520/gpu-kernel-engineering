#include "gemm_tensorcore/reference.hpp"

#include <cmath>
#include <stdexcept>

void gemm_tensorcore::reference_wmma_single(const __half *a, const __half *b,
                                            float *c) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "WMMA reference requires non-null A/B/C buffers");
    }

    for (int row = 0; row < wmma_m; ++row) {
        for (int column = 0; column < wmma_n; ++column) {
            float accumulator = 0.0F;
            for (int inner = 0; inner < wmma_k; ++inner) {
                const float a_value = __half2float(a[row * wmma_k + inner]);
                const float b_value = __half2float(b[inner * wmma_n + column]);
                accumulator = fmaf(a_value, b_value, accumulator);
            }
            c[row * wmma_n + column] = accumulator;
        }
    }
}
