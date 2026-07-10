#pragma once

#include <cstddef>

namespace gemm {

struct ErrorMetrics {
    double max_abs;
    double max_rel;
    std::size_t worst_index;
    float expected;
    float actual;
    bool finite;
};

ErrorMetrics compare(const float* expected, const float* actual, std::size_t count);

// Conservatively accepts only when all elements meet the absolute tolerance or
// all elements meet the relative tolerance; mixed allclose cases may be rejected.
// Throws std::invalid_argument when either tolerance is negative.
bool passes(const ErrorMetrics& metrics, double atol, double rtol);

}  // namespace gemm
