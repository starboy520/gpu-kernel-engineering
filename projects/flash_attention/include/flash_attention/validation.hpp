#pragma once

#include <cstddef>

namespace flash_attention {

struct ErrorMetrics {
    double max_abs;
    double max_rel;
    std::size_t worst_index;
    float expected;
    float actual;
    bool finite;
};

ErrorMetrics compare(const float *expected, const float *actual,
                     std::size_t count);
bool passes(const ErrorMetrics &metrics, double atol, double rtol);

} // namespace flash_attention