#pragma once

#include "gemm_tensorcore/kernel.hpp"

namespace gemm_tensorcore {

// Reference consumes the already-quantized FP16 values and accumulates in
// FP32, matching the G0/G1 numerical contract.
void reference_wmma_single(const __half *a, const __half *b, float *c);

// Rounds every positive dimension up to the next WMMA tile boundary.
// Throws std::invalid_argument for non-positive dimensions and
// std::overflow_error when rounding cannot be represented by int.
Problem pad_problem(Problem problem);

// Direct mathematical reference for A row-major and B col-major. Leading
// dimensions are measured in elements: lda is A's row stride, ldb is B's
// column stride, and ldc is C's row stride.
void reference_wmma_direct(const __half *a, int lda, const __half *b, int ldb,
                           float *c, int ldc, Problem problem);

} // namespace gemm_tensorcore
