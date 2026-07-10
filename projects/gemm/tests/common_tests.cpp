#include "gemm/reference.hpp"
#include "gemm/validation.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void check_invalid_argument(Function&& function, const std::string& message) {
    try {
        function();
        check(false, message + " did not throw std::invalid_argument");
    } catch (const std::invalid_argument&) {
    } catch (...) {
        check(false, message + " threw the wrong exception type");
    }
}

void test_reference_cpu_hand_result() {
    const float a[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float b[] = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    const float expected[] = {58.0F, 64.0F, 139.0F, 154.0F};
    float actual[4] = {};

    gemm::reference_cpu(a, b, actual, 2, 2, 3);

    for (std::size_t index = 0; index < 4; ++index) {
        check(actual[index] == expected[index],
              "reference_cpu hand result at index " + std::to_string(index));
    }
}

void test_reference_cpu_contract() {
    float value = 1.0F;

    check_invalid_argument(
        [&] { gemm::reference_cpu(nullptr, &value, &value, 1, 1, 1); },
        "reference_cpu null A");
    check_invalid_argument(
        [&] { gemm::reference_cpu(&value, nullptr, &value, 1, 1, 1); },
        "reference_cpu null B");
    check_invalid_argument(
        [&] { gemm::reference_cpu(&value, &value, nullptr, 1, 1, 1); },
        "reference_cpu null C");
    check_invalid_argument(
        [&] { gemm::reference_cpu(&value, &value, &value, 0, 1, 1); },
        "reference_cpu non-positive M");
    check_invalid_argument(
        [&] { gemm::reference_cpu(&value, &value, &value, 1, 0, 1); },
        "reference_cpu non-positive N");
    check_invalid_argument(
        [&] { gemm::reference_cpu(&value, &value, &value, 1, 1, 0); },
        "reference_cpu non-positive K");
}

void test_compare_worst_absolute_error() {
    const float expected[] = {1.0F, -2.0F, 4.0F};
    const float actual[] = {1.25F, -3.0F, 4.5F};

    const gemm::ErrorMetrics metrics = gemm::compare(expected, actual, 3);

    check(metrics.max_abs == 1.0, "compare max_abs");
    check(metrics.max_rel == 0.5, "compare max_rel");
    check(metrics.worst_index == 1, "compare worst_index");
    check(metrics.expected == -2.0F, "compare stored expected value");
    check(metrics.actual == -3.0F, "compare stored actual value");
    check(metrics.finite, "compare finite inputs");
}

void test_compare_non_finite_actual_values() {
    const float expected = 1.0F;
    const float non_finite_values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    const char* labels[] = {"NaN", "positive infinity", "negative infinity"};

    for (std::size_t index = 0; index < 3; ++index) {
        const gemm::ErrorMetrics metrics =
            gemm::compare(&expected, &non_finite_values[index], 1);
        check(!metrics.finite,
              std::string("compare marks actual ") + labels[index] + " non-finite");
    }
}

void test_compare_non_finite_expected_values() {
    const float actual = 1.0F;
    const float non_finite_values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    const char* labels[] = {"NaN", "positive infinity", "negative infinity"};

    for (std::size_t index = 0; index < 3; ++index) {
        const gemm::ErrorMetrics metrics =
            gemm::compare(&non_finite_values[index], &actual, 1);
        check(!metrics.finite,
              std::string("compare marks expected ") + labels[index] + " non-finite");
    }
}

void test_compare_contract() {
    const float value = 1.0F;

    check_invalid_argument([&] { gemm::compare(nullptr, &value, 1); },
                           "compare null expected");
    check_invalid_argument([&] { gemm::compare(&value, nullptr, 1); },
                           "compare null actual");
    check_invalid_argument([&] { gemm::compare(&value, &value, 0); },
                           "compare zero count");
}

void test_passes_rejects_non_worst_absolute_tolerance_violation() {
    const float expected[] = {1000.0F, 0.001F};
    const float actual[] = {1001.0F, 0.002F};
    const gemm::ErrorMetrics metrics = gemm::compare(expected, actual, 2);

    check(!gemm::passes(metrics, 0.0, 0.001),
          "passes rejects relative violation outside worst absolute element");
}

void test_passes_aggregate_criteria_and_contract() {
    const gemm::ErrorMetrics absolute_pass{0.1, 10.0, 0, 0.0F, 1.0F, true};
    check(gemm::passes(absolute_pass, 0.1, 0.01),
          "passes accepts by global absolute criterion");

    const gemm::ErrorMetrics relative_pass{10.0, 0.001, 0, 0.0F, 1.0F, true};
    check(gemm::passes(relative_pass, 0.1, 0.001),
          "passes accepts by global relative criterion");

    const gemm::ErrorMetrics aggregate_failure{0.2, 0.002, 0, 0.0F, 0.0F, true};
    check(!gemm::passes(aggregate_failure, 0.1, 0.001),
          "passes rejects when both global maxima exceed tolerances");

    const gemm::ErrorMetrics non_finite{0.0, 0.0, 0, 0.0F, 0.0F, false};
    check(!gemm::passes(non_finite, 1000.0, 1000.0),
          "passes rejects non-finite metrics");

    check_invalid_argument([&] { gemm::passes(non_finite, -0.1, 0.0); },
                           "passes negative atol");
    check_invalid_argument([&] { gemm::passes(non_finite, 0.0, -0.1); },
                           "passes negative rtol");
}

}  // namespace

int main() {
    test_reference_cpu_hand_result();
    test_reference_cpu_contract();
    test_compare_worst_absolute_error();
    test_compare_non_finite_actual_values();
    test_compare_non_finite_expected_values();
    test_compare_contract();
    test_passes_rejects_non_worst_absolute_tolerance_violation();
    test_passes_aggregate_criteria_and_contract();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All gemm_common tests passed\n";
    return 0;
}
