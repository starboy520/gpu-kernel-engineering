#include "gpu_kernel/validation.hpp"
#include "gpu_kernel/runner_utils.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_compare_and_passes() {
    const float expected[] = {1.0F, -2.0F, 0.0F};
    const float actual[] = {1.1F, -2.0F, 0.0F};
    const gpu_kernel::ErrorMetrics metrics =
        gpu_kernel::compare(expected, actual, 3);

    check(metrics.finite, "finite inputs stay finite");
    check(std::abs(metrics.max_abs - 0.1) < 1.0e-6,
          "maximum absolute error is recorded");
    check(metrics.worst_index == 0, "worst absolute index is recorded");
    check(gpu_kernel::passes(metrics, 0.11, 0.0),
          "absolute tolerance can accept output");
    check(!gpu_kernel::passes(metrics, 0.01, 0.01),
          "insufficient aggregate tolerances reject output");
}

void test_invalid_and_non_finite_inputs() {
    const float value = 1.0F;
    bool null_rejected = false;
    try {
        (void)gpu_kernel::compare(nullptr, &value, 1);
    } catch (const std::invalid_argument &) {
        null_rejected = true;
    }
    check(null_rejected, "null input is rejected");

    const float infinity = std::numeric_limits<float>::infinity();
    const gpu_kernel::ErrorMetrics metrics =
        gpu_kernel::compare(&value, &infinity, 1);
    check(!metrics.finite, "non-finite actual value is rejected");
    check(!gpu_kernel::passes(metrics, 1.0e9, 1.0e9),
          "non-finite metrics never pass");
}

void test_runner_leaf_utilities() {
    check(gpu_kernel::checked_multiply(3, 7, "test") == 21,
          "checked multiply returns product");

    bool overflow_rejected = false;
    try {
        (void)gpu_kernel::checked_multiply(
            std::numeric_limits<std::size_t>::max(), 2, "test");
    } catch (const std::overflow_error &) {
        overflow_rejected = true;
    }
    check(overflow_rejected, "checked multiply rejects overflow");

    check(gpu_kernel::parse_integer<int>("42", "--value") == 42,
          "integer parser accepts complete integer");
    bool invalid_rejected = false;
    try {
        (void)gpu_kernel::parse_integer<int>("42x", "--value");
    } catch (const std::invalid_argument &) {
        invalid_rejected = true;
    }
    check(invalid_rejected, "integer parser rejects trailing text");

    const char *arguments[] = {"runner", "--value", "17"};
    int index = 1;
    check(gpu_kernel::require_value(3, arguments, index, "--value") == "17",
          "required option value is returned");
    check(index == 2, "required option advances argument index");
}

} // namespace

int main() {
    test_compare_and_passes();
    test_invalid_and_non_finite_inputs();
    test_runner_leaf_utilities();
    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "All gpu_kernel_common tests passed\n";
    return 0;
}