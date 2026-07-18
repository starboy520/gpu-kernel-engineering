#include "gemm_tensorcore/reference.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

int round_up_to_tile(int value) {
    if (value <= 0) {
        throw std::invalid_argument(
            "WMMA padding requires positive dimensions");
    }
    constexpr int tile = gemm_tensorcore::wmma_m;
    if (value > std::numeric_limits<int>::max() - (tile - 1)) {
        throw std::overflow_error("WMMA padded dimension overflow");
    }
    return ((value + tile - 1) / tile) * tile;
}

} // namespace

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

gemm_tensorcore::Problem
gemm_tensorcore::pad_problem(gemm_tensorcore::Problem problem) {
    return {round_up_to_tile(problem.m), round_up_to_tile(problem.n),
            round_up_to_tile(problem.k)};
}

void gemm_tensorcore::reference_wmma_direct(const __half *a, int lda,
                                            const __half *b, int ldb, float *c,
                                            int ldc, Problem problem) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "Direct WMMA reference requires non-null A/B/C buffers");
    }
    if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
        throw std::invalid_argument(
            "Direct WMMA reference requires positive dimensions");
    }
    if (lda < problem.k || ldb < problem.k || ldc < problem.n) {
        throw std::invalid_argument(
            "Direct WMMA reference received an invalid leading dimension");
    }

    for (int row = 0; row < problem.m; ++row) {
        for (int column = 0; column < problem.n; ++column) {
            float accumulator = 0.0F;
            for (int inner = 0; inner < problem.k; ++inner) {
                const float a_value = __half2float(a[row * lda + inner]);
                const float b_value = __half2float(b[inner + column * ldb]);
                accumulator = fmaf(a_value, b_value, accumulator);
            }
            c[row * ldc + column] = accumulator;
        }
    }
}
