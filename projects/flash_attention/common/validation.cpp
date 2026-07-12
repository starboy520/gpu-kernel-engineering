#include "flash_attention/validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flash_attention {

ErrorMetrics compare(const float* expected, const float* actual,
                     std::size_t count) {
    if (expected == nullptr || actual == nullptr) {
        throw std::invalid_argument("compare requires non-null inputs");
    }
    if (count == 0) {
        throw std::invalid_argument("compare requires at least one element");
    }

    ErrorMetrics metrics{0.0, 0.0, 0, expected[0], actual[0], true};
    for (std::size_t index = 0; index < count; ++index) {
        const double expected_value = static_cast<double>(expected[index]);
        const double actual_value = static_cast<double>(actual[index]);
        double abs_error = std::abs(actual_value - expected_value);
        double rel_error =
            abs_error / std::max(std::abs(expected_value), 1.0e-12);

        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            metrics.finite = false;
            if (std::isnan(abs_error)) {
                abs_error = std::numeric_limits<double>::infinity();
            }
            if (std::isnan(rel_error)) {
                rel_error = std::numeric_limits<double>::infinity();
            }
        }

        if (abs_error > metrics.max_abs) {
            metrics.max_abs = abs_error;
            metrics.worst_index = index;
            metrics.expected = expected[index];
            metrics.actual = actual[index];
        }
        metrics.max_rel = std::max(metrics.max_rel, rel_error);
    }
    return metrics;
}

bool passes(const ErrorMetrics& metrics, double atol, double rtol) {
    if (atol < 0.0 || rtol < 0.0) {
        throw std::invalid_argument("passes requires non-negative tolerances");
    }
    return metrics.finite &&
           (metrics.max_abs <= atol || metrics.max_rel <= rtol);
}

}  // namespace flash_attention